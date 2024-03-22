import re
import copy
import torch as tor

# same folder
from . import NonLinearity as NL
from . import Parser_0_2 as Parser

from .. import HelperFuncs as HF # Folder above

Device = HF.Set_Tensortype_And_Device()

########################################################################################################################
#####                                            Processing functions                                              #####
########################################################################################################################
# This sections contains everything involving the transformation from the parsed expression a NARMAX lambda can can be evaluated.
# Those are only called by the SymbolicOscillator CTor and were initially part of it.
# They were separated from the class to make the unit tests simpler, since no default object exists to prgressively call them.

# ################################################### Create EvalStr ###################################################
def ParsedReg2EvalStr( RegStr: Parser.ParsedReg, VarName2Idx: dict, NonLinName2Idx: dict, OutputVarName = 'y' ):
  """
  Transform the expression into a string that can be evaluated by the python interpreter.
  Spam everything with parentheses to be sure the evaluation order is correct.
  """

  OutStr = "" # Covers the case of RegStr.FuncName is None
  if ( RegStr.FuncName is not None ):
    if ( RegStr.FuncName in ["~/", "1/"] ): # Those exact names: fractional/denominator with no supplementary Non-lin
      OutStr += RegStr.FuncName # no further processing needed
    elif ( ( len( RegStr.FuncName ) > 2 ) and ( RegStr.FuncName[:2] in ["~/", "1/"] ) ): # fractional/denominator non-lin
      OutStr += RegStr.FuncName[:2] + f"NonLinList[{ NonLinName2Idx[RegStr.FuncName[2:]] }].get_f()" # look-up without, but prepend
    else: # Non-fractional aka Numerator expression
      OutStr += f"NonLinList[{ NonLinName2Idx[RegStr.FuncName] }].get_f()"

  OutStr += "("
  for i in range( len( RegStr.SubExpressions ) ):
    if ( RegStr.SubExpressions[i].VarName is not None ):
      TempSubExpr = copy.deepcopy( RegStr.SubExpressions[i] )
      if ( RegStr.SubExpressions[i].VarName == OutputVarName ): TempSubExpr.VarName = "OutVec" # reserve name to allow user to have y as variable
      else: TempSubExpr.VarName = f"Data[{ VarName2Idx[TempSubExpr.VarName] }]"
      OutStr += f"({ TempSubExpr })"
    
    else: OutStr += f"({ RegStr.SubExpressions[i] })" # don't copy and process the variable if const or y

    if ( i != len( RegStr.Operators ) ): OutStr += " " + RegStr.Operators[i] + " " # one less oàperator than variables

  return ( OutStr + ")" )


# ############################################### Verify the Expression ##############################################
def VerifyParsedReg( Regressor: Parser.ParsedReg, NonLinNames, InputVarNames, OutputVarName = 'y' ):
  """Verifies that:
  - All used variables names are declared (contained into InputVarNames)
  - All used non-linearity names are declared (contained into NonLinNames)
  - No non-negative lags of the Output Variable are used
  
  ### Inputs:
  - `RegStr`: (ParsedReg) containing the RegressorStr objects
  - `NonLinNames`: (list of NonLin objects) containing the non-linearity objects
  - `InputVarNames`: (list of str) containing the input variable names
  """

  # Check if the needed non-linearity exist
  if ( Regressor.FuncName is not None ):
    CurrentName = Regressor.FuncName
    if ( ( CurrentName[:2] == '~/' ) or ( CurrentName[:2] == '1/' ) ): CurrentName = CurrentName[2:] # fractions are never stored as separate functions
    if ( ( CurrentName != '' ) and (CurrentName not in NonLinNames)): # empty string if fraction/denominator term without non-lin
      raise ValueError( f"Non-linearity '{ CurrentName }' is not a declared Non-linearity" )

  for subexp in Regressor.SubExpressions:

    # Check if all needed variables exist
    if ( subexp.VarName is None ): continue # Skip constants

    if ( subexp.VarName not in InputVarNames ): raise ValueError( f"Variable '{ subexp.VarName }' is not a declared Variable" )

    if ( subexp.VarName in OutputVarName ):
      if ( subexp.Lag is None ): raise ValueError( f"No lag for output variable '{ OutputVarName }', as found in the expression { subexp } is not supported (non-causality)" )
      if( subexp.Lag is not None ):
        if ( subexp.Lag >= 0 ): raise ValueError( f"Positive lag for output variable '{ OutputVarName }' as found in expression { subexp } is not supported (non-causality)" )


# ############################################## Make Buffer Start System ##############################################
def Make_BufferStartSystem( Expr ):

  Output = re.sub( r'Data\[(\d+)\]\[k(([-+])(\d+))?\]', # Regexp recognizing Data = Input regressors
                  lambda match: f"Toggle(0, { int( match.group(1) ) }, k{ match.group(2) if match.group(2) is not None else '' }, Data)",
                  Expr
                )
  
  Output = re.sub( r'OutVec\[k(([-+])(\d+))?\]', # Regexp recognizing OutVec = Output regressors
                  lambda match: f"Toggle(1, None, k{ match.group(1) if match.group(1) is not None else '' }, Data)",
                  Output # Overwrite
                )
  
  return ( Output )


# ################################################ Make System Lambdas ###############################################
def Make_SystemLambdas( InputVarNames, OutputVarName, NonLinName2Idx, VarName2Idx, RegStrList ):
  for reg in RegStrList:
    VerifyParsedReg( reg, NonLinName2Idx.keys(), InputVarNames, OutputVarName ) # Are all required variables & nonLins passed + no sus lags?

  NumeratorExpr = ''
  DenominatorExpr = '' # stays an empty string for non-rational NARMAXes

  for idx, reg in enumerate( RegStrList ):
    if ( reg.FuncName is None ): # Automatically a numerator term, als '~/' is considered a function name
      NumeratorExpr += f"theta[{ idx }]*" + ParsedReg2EvalStr( reg, VarName2Idx, NonLinName2Idx ) + ' + '
  
    else: # Potentially denominator term
      if ( reg.FuncName[:2] == '~/' ):
        DenominatorExpr += f"theta[{ idx }]*" + ParsedReg2EvalStr( reg, VarName2Idx, NonLinName2Idx ).replace( '~/', '' )  + ' + '
      else:
        NumeratorExpr   += f"theta[{ idx }]*" + ParsedReg2EvalStr( reg, VarName2Idx, NonLinName2Idx )  + ' + '
    
# remove the last ' + '
  NumeratorExpr = NumeratorExpr[:-3]
  if ( DenominatorExpr != '' ): DenominatorExpr = "1 + " + DenominatorExpr[:-3] # necessary, the way the arbo generates rational systems

# Create System lambdas. Bufferstart has no OutVec since accessed as class member
  if ( DenominatorExpr ):
    System_Main =        eval( f"lambda k, theta, Data, OutVec, NonLinList: ({ NumeratorExpr }) / ({ DenominatorExpr })" )
    System_BufferStart = eval( f"lambda k, theta, Data, NonLinList, Toggle: ({ Make_BufferStartSystem( NumeratorExpr ) }) / "
                                                                          f"({ Make_BufferStartSystem( DenominatorExpr ) })"
                            )
  else: # Denom is ''
    System_Main        = eval( f"lambda k, theta, Data, OutVec, NonLinList: { NumeratorExpr }" )
    System_BufferStart = eval( f"lambda k, theta, Data, NonLinList, Toggle: { Make_BufferStartSystem( NumeratorExpr ) }" ) # vars are row-wise like the user input

  return ( System_BufferStart, System_Main )


########################################################################################################################
#####                                             SYMBOLIC OSCILLATOR                                              #####
########################################################################################################################
class SymbolicOscillator:

  # ####################################################### CTor #######################################################
  def __init__( self, InputVarNames, NonLinearities, ExprList, theta, OutputVarName = 'y', dtype = tor.float64, device = Device ):
    """Generates the Regressor strings contained in the RegStr
    
    ### Inputs:
    - `InputVarNames`: (list of str) containing the regressor names
    - `NonLinearities`: (list of rFOrLSR.NonLinearity) containing the non-linearity objects
    - `ExprList`: (list of str) containing the RegressorStr objects
    - `theta`: ((DsData.shape[1] + len( ExprList)),)-shaped torch tensor) containing the regression Coefficients
    - `OutputVarName`: (str) containing the name used to denote the output variable in the passed strings
    - `dtype`: (torch.dtype = torch.float64) containing the type of the output variable
    - `device`: (torch.device = rFOrLSr.device) containing the device of the output buffer
    """
    # --------------------------------------------------- Input checks ---------------------------------------------------
    if ( len( InputVarNames ) == 0 ):  raise ValueError( "No Input variables were declared" )
    for i in range( len( InputVarNames ) ):
      if ( not isinstance( InputVarNames[i], str ) ): raise ValueError( f"InputVarNames[{ i }] is not of type 'str'" )
    
    if ( len( NonLinearities ) == 0 ): raise ValueError( "No Non-linearities were declared" )
    for i in range( len( NonLinearities ) ):
      if ( not isinstance( NonLinearities[i], NL.NonLinearity ) ): raise ValueError( f"NonLinearities[{ i }] is not of type 'rFOrLSR.NonLinearity'" )

    if ( len( ExprList ) == 0 ):     raise ValueError( "No Regressors were declared" )
    for i in range( len( ExprList ) ):
      if ( not isinstance( ExprList[i], str ) ): raise ValueError( f"RegStrList[{ i }] is not a string" )
    
    if ( not isinstance( theta, tor.Tensor ) ): raise ValueError( f"theta must be of type 'torch.Tensor'" )

    if ( not isinstance( OutputVarName, str ) ): raise ValueError( f"OutputVarName is not of type 'str'" )

    if ( not isinstance( dtype, tor.dtype ) ): raise ValueError( f"dtype is not of type 'torch.dtype'" )

    # Just Poker that the passed device is valid, torch will certainly complain otherwise

    # -------------------------------------------------- Data Storage --------------------------------------------------
    # The Class stores only the data needed to oscillate, everything else is discarded

    # Processing parameters
    self.theta = theta
    self.NonLinearities = NonLinearities
    self.nExpressions = len( ExprList )
    self.nInputVars = len( InputVarNames ) if ( OutputVarName not in InputVarNames ) else len( InputVarNames ) - 1 # Don't count the output variable, it has own buffer

    # Output Data parameters
    self.dtype = dtype
    self.device = device

    # Systems Lambdas
    self.System_BufferStart = None # Lambda containing the system run in the left buffer border part (buffer start)
    self.System_Main = None # Lambda containing the system run in the non-border buffer part

    # MaxLags data for the current system. Uninitialized since strings not yet parsed
    self.MaxNegLag = 0 # Maximum negative delay for input data
    self.MaxPosLag = 0 # To support acausal systems (in non-Output terms, since for Outputs VerifyExpression throws for lags >= 0)
    self.MaxOutputLag = 0 # Maximum negative delay for the output data
    
    # Internal storage for in/out past states to bridge buffer ends
    # Note: one can't use negative indices for the OutVec in python, since the user might use lags greater than the buffer or resize the buffer, which reinitializes it
    self.InputStorage = None # for any variable indexed by k-j with j > 0
    self.OutputStorage = None # for OutVec[k-j] with j > 0
    
    self.OutVec = tor.zeros( 8, dtype = self.dtype, device = self.device ) # arbitrary length as placeholder: only known when user calls Oscillate()

    # --------------------------------------- Convertion Dicts and Reg validation --------------------------------------
    NonLinName2Idx = {} # key: name, value: index in NonLinearities
    for idx in range( len( NonLinearities ) ): NonLinName2Idx[ NonLinearities[idx].get_Name() ] = idx

    VarName2Idx = {} # key: name, value: index in Data
    for idx in range( len( InputVarNames ) ): VarName2Idx[ InputVarNames[idx] ] = idx
    
    OutVecError = "The name 'OutVec' is reserved for internal processing, plase rename that "
    if ( "OutVec" in VarName2Idx.keys() ):    raise ValueError( OutVecError + "Variable" )
    if ( "OutVec" in NonLinName2Idx.keys() ): raise ValueError( OutVecError + "NonLinearity" )

    RegStrList = [ Parser.ExpressionParser( expr ) for expr in ExprList ] # Parsed Regressor Objects

    # ----------------------------------------------- Generate Expressions -----------------------------------------------
    self.System_BufferStart, self.System_Main = Make_SystemLambdas( InputVarNames, OutputVarName, NonLinName2Idx, VarName2Idx, RegStrList )

    # ----------------------------------------- Find Maxlags and create buffers ----------------------------------------
    for reg in RegStrList:
      for se in reg.SubExpressions:
        if ( ( se.Lag is None ) or ( se.Lag == 0 ) ): continue # Coeff which has no lag, also se.VarName is None

        if ( se.VarName == OutputVarName ): self.MaxOutputLag = max( self.MaxOutputLag, abs( se.Lag ) )    
        else: # Input Data lags
          if   ( se.Lag < 0 ): self.MaxNegLag = max( self.MaxNegLag, abs( se.Lag ) )
          elif ( se.Lag > 0 ): self.MaxPosLag = max( self.MaxPosLag, se.Lag )

    self.OutputStorage = tor.zeros( self.MaxOutputLag, dtype = self.dtype, device = self.device )
    self.InputStorage = tor.zeros( ( self.nInputVars, self.MaxNegLag ), dtype = self.dtype, device = self.device ) 


  # ################################################### theta setter ###################################################
  def set_theta( self, theta ):
    if ( theta.shape[0] != self.nExpressions ): raise ValueError( f"theta has wrong dimension, expected { self.nExpressions }" )
    self.theta = theta

  
  # ##################################################### Oscillate ####################################################
  def Toggle( self, DataOrOutVec, VarNumber, k, Data ):
    """ Helper function which toggles between the stored internal system state for k < 0 and incomming data for k >= 0.
    VarNumber is the index of the variable in the Data array, Later for -MO-systems it will also be used for OutVec
    """
    if ( DataOrOutVec == 0 ): # 0 for Input
      if ( k >= 0 ): return Data[ VarNumber ][ k ] # normal indexing since k >= 0
      else:          return self.InputStorage[ VarNumber, k ] # k is negative: indexing from the end

    elif ( DataOrOutVec == 1 ): # 1 forOoutput, same as above but without variable dispaching (currently only one output supported)
      if ( k >= 0 ): return self.OutVec[ k ]
      else:          return self.OutputStorage[ k ]

    else: raise ValueError( "Internal Error, please report this bug: 'DataOrOutVec is neither Data nor OutVec'" )


  # ##################################################### Oscillate ####################################################
  def Oscillate( self, Data, theta = None, DsData = None ):
    """ Function applying the NARMAX-system to the input data.
    DsData can is allowed to change the number of columns it has (if correctly mirrored in theta), since the system has no concept of it.

    ### Inputs:
    - `Data`: (iterable of 1D-torch.tensors having all the same length) containing the input variable vectors
    - `theta`: (optional (DsData.shape[1] + len( ExprList)),)-shaped torch.tensor) containing the regression Coefficients allowing to modulate theta
    - `DsData`: (optional 1D-tensor) containing any signal to be directly (no scaling, processing or storage) injected into the system
    """
    # -------------------------------------------- System Parameter Update ---------------------------------------------
    if ( DsData is not None ):
      if ( DsData.ndim != 1 ): raise ValueError( "DsData must be 1D" )
      if ( DsData.shape[0] != Data[0].shape[0] ): raise ValueError( f"DsData's dimension doesn't equal that of Data, being { Data.shape[0] }" )
    
    if ( theta is not None ): self.set_theta( theta )

    if ( self.OutVec.shape[0] != Data[0].shape[0] ): # stored to avoid re-allocating, update if necessary
      self.OutVec = tor.zeros( Data[0].shape[0], dtype = self.dtype, device = self.device ) # reinitialize since not used in state-storage

    if ( self.MaxNegLag + self.MaxPosLag > Data[0].shape[0] ):
      raise ValueError( f"Input buffer-size is smaller than the system's total lag range of { self.MaxNegLag + self.MaxPosLag }. Not illegal but not supported yet." )

    # --------------------------------------------------- Processing ---------------------------------------------------
    if ( DsData is not None ): self.OutVec = DsData.clone() # pre-allocate for performance
    else:                      self.OutVec = tor.zeros_like( Data[0] )

    for k in range( 0, self.MaxNegLag ): # Buffer start procedure, with dispatch to internal state via toggle
      self.OutVec[k] += self.System_BufferStart( k, self.theta, Data, self.NonLinearities, self.Toggle )

    for k in range( self.MaxNegLag, Data[0].shape[0] - self.MaxPosLag ): # Fully swung-in state
      self.OutVec[k] += self.System_Main( k, self.theta, Data, self.OutVec, self.NonLinearities )

    # --------------------------------------------- Internal State Update ----------------------------------------------
    for input in range ( len( Data ) ): # Must iterate one by one since Data is an arbirtary iterable not necessarily a 2D Tensor
      if ( Data[input].ndim != 1 ): raise ValueError( f"Input Data must be 1D. The { input }-th input is not" )
      self.InputStorage[ input ] = Data[ input ][ -self.MaxNegLag : ].clone()
    
    self.OutputStorage = self.OutVec[ -self.MaxOutputLag : ].clone() # keep last outputs for next buffer
    
    return ( self.OutVec )


########################################################################################################################
#####                                                  UNIT TESTS                                                  #####
########################################################################################################################

if ( __name__ == '__main__' ):
  pass