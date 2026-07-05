import torch as tor

# Rest of the Lib: "from ."" is same Folder and ".Name" is subfolder import
from .Classes.NonLinearity import NonLinearity # for typechecking
from .Classes.SymbolicOscillator_0_4 import SymbolicOscillator # for Default Validation

from typing import Optional, Sequence, Any
from numpy.typing import NDArray
from numpy import float64, int64, str_ # for typechecking only no numpy dependency here

############################################################################### Compute NARMAX Output ##############################################################################
def InitAndComputeBuffer( Model: SymbolicOscillator, y: tor.Tensor, Data: Sequence[ tor.Tensor ], DsData: Optional[ tor.Tensor ] = None
                        ) -> tor.Tensor:
  '''
  Initialises the NARMAX model's internal state and generates its output from the given data, so that the output exactly matches
  the true system output y.

  Parameters
  ----------
  Model : SymbolicOscillator
      The compiled model.
  y : torch.Tensor (1D)
      True output sequence (needed for the initial state).
  Data : sequence of 1D torch.Tensors
      Input variable sequences, one per input channel.
  DsData : torch.Tensor, optional
      Additive signal directly injected into the system (no lag).
      Must have the same length as the input tensors.

  Returns
  -------
  yHat : torch.Tensor
      Model output, same shape/device as `y`.
  '''

  if ( ( Model.get_MaxPositiveInputLag() ) > 0 ): raise RuntimeError( "InitAndComputeBuffer does not currently support models with positive input lags (non‑causal terms)." )

  # ------------------------------------------------------------------  Input sanity checks # ------------------------------------------------------------------
  if ( len( Data ) != Model.get_nInputVars() ):
    raise ValueError( f"Number of input tensors ({ len( Data ) }) does not match model's number of input variables ({ Model.get_nInputVars() })." )

  for i, inp in enumerate( Data ):
    if ( inp.ndim != 1 ): raise ValueError( f"Input Data[{ i }] must be 1‑dimensional." )
    if ( inp.shape[ 0 ] != y.shape[ 0 ] ):
      raise ValueError( f"Data[{ i }] length ({ inp.shape[ 0 ] }) differs from  output length ({ y.shape[ 0 ] }). All sequences must have the same length." )

  if ( DsData is not None ):
    if ( DsData.ndim != 1 ):              raise ValueError( "DsData must be 1‑dimensional." )
    if ( DsData.shape[ 0 ] != y.shape[ 0 ] ): raise ValueError( "DsData must have the same length as y and Data." )

  # 3. Determine the initial window
  StartIdx: int = max( Model.get_MaxInputLag(), Model.get_MaxOutputLag() ) # Loop start = max( qx, qy ) as usual

  if ( StartIdx > y.shape[ 0 ] ): raise ValueError( "Sequence too short to cover the required lags." )

  # 4. Set internal state using the immediately preceding window
  # Output storage: y[k-j] for j in 1..qy, i.e., the qy samples just before StartIdx.
  Model.set_OutputStorage( y[ StartIdx - Model.get_MaxOutputLag() : StartIdx ].clone() )

  # Input storage: for each channel, the qx samples just before StartIdx.
  Model.set_InputStorage( tor.vstack( [ input[ StartIdx - Model.get_MaxInputLag() : StartIdx ] for input in Data ] ) )

  # 5. Build output buffer
  yHat: tor.Tensor = tor.zeros_like( y )

  # The first `StartIdx` samples are taken directly from the true output, because the model cannot yet compute them (missing past values).
  yHat[ : StartIdx ] = y[ : StartIdx ].clone()

  # Slices of Data (and DsData) that the model will process.
  DataSlice = [ inp[ StartIdx : ] for inp in Data ]
  DsSlice = DsData[ StartIdx : ] if DsData is not None else None

  # Run the model on the remaining part and move the result to the same device as `y` to avoid device mismatch errors.
  yHat[ StartIdx : ] = Model.Oscillate( DataSlice, DsData = DsSlice ).to( y.device )

  return yHat

############################################################################ Default Validation procedure ##########################################################################
def DefaultValidation( theta: tor.Tensor, L: NDArray[ int64 ], ERR: NDArray[ float64 ], RegNames: NDArray[ str_ ],
                      ValData: dict[ str, Any ], DcFilterIdx: Optional[ NDArray[ int64 ] ] = None ) -> float:
  '''
  Default Validation function based on time domain relative MAE.

  ### Input
  - `theta`: ( (nr,)-shaped float tor.tensor ) containing the estimated regression coefficients
  - `L`: ( (nr,)-shaped int nd-array ) containing the regressors indices
  - `ERR`: ( (nr,)-shaped float nd-array ) containing the regression's ERR ( not used here but placeholder to adhere to AOrLSR standard )
  - `RegNames`: ( (nr,)-shaped str nd-array ) containing only the needed regressor names, serves as system equation look-up
  - `DcFilterIdx`: ( (nr,)-shaped int nd-array ) containing the regressors indices to be filtered, unused by this function but standard in the API

  - `ValData`: ( dict ) containing at least the validation data to be passed to Dc's Ctors:
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

  if ( not isinstance( ValData[ "y" ], list ) ): raise AssertionError( "ValData's 'y' entry is expected to be a list of float tor.Tensors" )

  # Data entry validation
  if ( ValData[ "Data" ] is None ): raise AssertionError( "ValData's 'Data' entry is None" )
  if ( not isinstance( ValData[ "Data" ], list ) ): raise AssertionError( "ValData's 'Data' entry is expected to be a list" )
  if ( len( ValData[ "Data" ] ) == 0 ): raise AssertionError( "ValData's 'Data' entry is empty, there is thus nothing to validation against" )
  if ( len( ValData[ "Data" ] ) != len( ValData[ "y" ] ) ): raise AssertionError( "ValData's 'Data' and 'y' lists should have the same length" )

  for val in range( len( ValData[ "Data" ] ) ): # iterate over all passed Data tuples
    # Validate the y entry for this validation set
    if ( not isinstance( ValData[ "y" ][ val ], tor.Tensor ) ): raise AssertionError( f"ValData's 'y' entry at index { val } is not a torch.Tensor" )
    if ( ValData[ "y" ][ val ].ndim == 0 ): raise AssertionError( f"ValData's 'y' entry at index { val } is a scalar tensor, must be at least 1-D" )

    # Validate the Data sub‑list
    ValData[ "Data" ][ val ] = ValData[ "Data" ][ val ]
    if ( not isinstance( ValData[ "Data" ][ val ], ( list, tuple ) ) ): raise AssertionError( f"ValData's 'Data' entry at index { val } is not a list/tuple" )
    if ( len( ValData[ "Data" ][ val ] ) == 0 ): raise AssertionError( f"ValData's 'Data' entry at index { val } is empty; the model expects at least one input tensor" )

    for reg in ValData[ "Data" ][ val ]: # Data entries contain unknown number of regressors in the MISO
      if ( not isinstance( reg, tor.Tensor ) ):          raise AssertionError( "ValData's 'Data' entry is expected to be a list of 2D-tuples of float tor.Tensors" )
      if ( ValData[ "y" ][ val ].shape[ 0 ] != reg.shape[ 0 ] ): raise AssertionError( f"ValData's 'Data' { val }-th entry's tensor has not the same length as the { val }-th 'y' entry" )

  # NonLinearities entry validation
  if ( not isinstance( ValData[ "NonLinearities" ], list ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )
  for func in ValData[ "NonLinearities" ]: # iterate over all passed NonLinearities
    if ( not isinstance( func, NonLinearity ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )

  # ------------------------------------------- Validation loop preparations -------------------------------------------
  if ( "OutputVarName" not in ValData.keys() ): OutputVarName: str = "y" # default if not passed
  else:                                         OutputVarName: str = ValData[ "OutputVarName" ]

  # Note: DcFilterIdx is accepted for API compatibility but not used inside this function.
  _ = DcFilterIdx

  Error: float = 0.0 # Accumulated relative error
  Model: SymbolicOscillator = SymbolicOscillator( ValData[ "InputVarNames" ], ValData[ "NonLinearities" ], RegNames, theta, OutputVarName )

  for val in range( len( ValData[ "Data" ] ) ): # iterate over all passed Data iterables (Validations)
    yHat: tor.Tensor = InitAndComputeBuffer( Model, ValData[ "y" ][ val ], ValData[ "Data" ][ val ] ) # Set internal state and compute output
    # Compute relative MAE with protection against division by zero
    denom = tor.mean( tor.abs( ValData[ "y" ][ val ] ) )
    if ( denom == 0.0 ): denom = + 1e-8 # If the target is all zeros, add a tiny epsilon to avoid infinity;

    Error += tor.mean( tor.abs( ValData[ "y" ][ val ] - yHat ) / denom ).item()

  return ( Error / len( ValData[ "Data" ] ) ) # normalised by number of validations  ( not necessary for AOrLSR but printed for the user )
