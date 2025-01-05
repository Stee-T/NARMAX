import torch as tor

# Rest of the Lib: "from ."" is same Folder and ".Name" is subfolder import
from .Classes.NonLinearity import NonLinearity # for typechecking
from .Classes.SymbolicOscillator_0_3 import SymbolicOscillator # for Default Validation

# ############################################ Compute NARMAX Output #############################################
def InitAndComputeBuffer( Model, y, Data ):
  '''Helper function initializing the NARMAX model and generating its output from the passed data.'''

  StartIdx = max( Model.get_MaxNegOutputLag(), Model.get_MaxNegInputLag() ) # essentially q = max(qx, qy) as usual

  Model.set_OutputStorage( y[ : Model.get_MaxNegOutputLag() ].clone() ) # set previous y[k-j] states
  Model.set_InputStorage( tor.vstack( [ input[ : Model.get_MaxNegInputLag() ] for input in Data ] ) ) # set previous phi[k-j] states

  yHat = tor.zeros_like( y )
  yHat[ :StartIdx ] = y[ : StartIdx ].clone() # take solution samples, where Model hasn't got all data. Avoids init-Error spikes
  yHat[ StartIdx: ] = Model.Oscillate( [ input[ StartIdx: ] for input in Data ] )
  
  return ( yHat )

# ############################################ Default Validation procedure #############################################
def DefaultValidation( theta, L, ERR, RegNames, ValData, DcFilterIdx = None ):
  '''
  Default Validation function based on time domain MAE.
  
  ### Input
  - `theta`: ( (nr,)-shaped float nd-array ) containing the estimated regression coefficients
  - `L`: ( (nr,)-shaped int nd-array ) containing the regressors indices
  - `ERR`: ( (nr,)-shaped float nd-array ) containing the regression's ERR ( not used here but placeholder to adhere to AOrLSR standard )
  - `RegNames`: ( (nr,)-shaped str nd-array ) containing only the needed regressor names, serves as system equation look-up
  - `DcFilterIdx`: ( (nr,)-shaped int nd-array ) containing the regressors indices to be filtered, unused by this function but standard in the API
  
  - `ValData`: ( dict ) containing the validation data to be passed to Dc's Ctors:
    → `ValData["y"]`: ( list of float torch.Tensors ) containing the system responses
    → `ValData["Data"]`: ( list of iterables of float torch.Tensors ) containing the data to be passed to the standard CTors ( Lagger, Expander, NonLinearizer )
    → `ValData["InputVarNames"]`: ( list of str ) containing the variable names as passed to Lagger, Expander, Nonlinearizer, so not all regressor names
    → `ValData["NonLinearities"]`: ( list of pointer to functions ) containing the regression's transformations to be passed to RegressorTransform
    → `ValData["OutputVarName"]`: ( optional - str ) containing the name of the output variable
  
  ### Output
  -`Error`: ( float ) containing the passed model's validation error on the validation set
  '''
  
  # ---------------------------------------  Bullshit prevention ----------------------------------------------------
  if ( not isinstance( ValData, dict ) ): raise AssertionError( "The passed ValData datastructure is not a dictionary as expected" )
  
  for var in [ "y", "InputVarNames", "Data", "NonLinearities" ]:
    if ( var not in ValData.keys() ): raise AssertionError( f"The validation datastructure contains no '{ var }' entry" )

  if ( not isinstance( ValData["y"], list ) ): raise AssertionError( "ValData's 'y' entry is expected to be a list of float torch.Tensors" )

  # Data entry validation
  if ( ( ValData["Data"] == [] ) or ( ValData["Data"] is None ) ):   raise AssertionError( "ValData's 'Data' entry is empty, there is thus no validation that can be performed" )
  if ( not isinstance( ValData["Data"], list ) ):         raise AssertionError( "ValData's 'Data' entry is expected to be a list" )
  if ( len ( ValData["Data"] ) != len ( ValData["y"] ) ): raise AssertionError( "ValData's 'Data' and 'y' lists should have the same length" )

  for val in range( len( ValData["Data"] ) ): # iterate over all passed Data tuples
    for reg in ValData["Data"][val]: # Data entries contain unknown number of regressors in the MISO
      if ( not isinstance( reg, tor.Tensor ) ): raise AssertionError( "ValData's 'Data' entry is expected to be a list of 2D-tuples of float torch.Tensors" )
      if ( ValData["y"][val].shape[0] != reg.shape[0] ): raise AssertionError( f"ValData's 'DsData' { val }-th entry has not the same length as the { val }-th 'y' entry" )

  # NonLinearities entry validation
  if ( not isinstance( ValData["NonLinearities"], list ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )
  for func in ValData["NonLinearities"]: # iterate over all passed NonLinearities
    if ( not isinstance( func, NonLinearity ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )
          
  # ------------------------------------------- Validation loop preparations -------------------------------------------
  if ( "OutputVarName" not in ValData.keys() ): OutputVarName = "y" # default if not passed
  else:                                         OutputVarName = ValData["OutputVarName"]

  Error = 0 # Total relative error
  Model = SymbolicOscillator( ValData["InputVarNames"], ValData["NonLinearities"], RegNames, theta, OutputVarName )

  for val in range( len( ValData["Data"] ) ): # iterate over all passed Data iterables (Validations)
    yHat = InitAndComputeBuffer( Model, ValData["y"][val], ValData["Data"][val] ) # Set internal state and compute output
    Error += tor.mean( tor.abs( ValData["y"][val] - yHat ) / tor.mean( tor.abs( ValData["y"][val] ) ) ) # relative MAE
    
  return ( Error.item() / len( ValData["Data"] ) ) # norm by the number of validations ( not necessary for AOrLSR but printed for the user )