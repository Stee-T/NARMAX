import re
import copy
import torch as tor

# same folder
from . import NonLinearity as NL
from . import Parser_0_2 as Parser

from typing import Optional, Sequence, Callable

def ScopeLimitHF() -> str: # Here to clarify that HF is only used for the device selection
  from .. import HelperFuncs # Parent folder
  return HelperFuncs.Set_Tensortype_And_Device()

Device: str = ScopeLimitHF() # Done here to avoid calling it twice (here and in the main __init__.py)

####################################################################################################################################################################################
#####                                                                          Processing functions                                                                            #####
####################################################################################################################################################################################
# This sections contains everything involving the transformation from the parsed expression a NARMAX lambda can can be evaluated.
# Those are only called by the SymbolicOscillator CTor and were initially part of it.
# They were separated from the class to make the unit tests simpler, since no default object exists to prgressively call them.

# ################################################################################# Create EvalStr #################################################################################
def ParsedReg2EvalStr( RegStr: Parser.ParsedReg, InputVarName2Idx: dict[ str, int ], NonLinName2Idx: dict[ str, int ], OutputVarName: str = 'y' ) -> tuple[ str, bool ]:
  '''Transform the expression into a string that can be evaluated by the python interpreter.
  Spam everything with parentheses to be sure the evaluation order is correct.

  ### Inputs:
  - `RegStr`: (ParsedReg) the parsed regressor expression
  - `InputVarName2Idx`: (dict) mapping from input variable name to index in Data
  - `NonLinName2Idx`: (dict) mapping from non-linearity name to index
  - `OutputVarName`: (str) name of the output variable

  ### Outputs:
  - `OutStr`: (str) evaluation-ready string expression
  - `is_AR`: (bool) whether the expression is auto-regressive
  '''
  is_AR: bool = False # Flag tracking if the current regressor contains the output varaible making it auto-regressive

  # -------------------------------- A. Function name / constant coefficient handling
  OutStr: str = "" # Covers the case of RegStr.FuncName is None
  if ( RegStr.FuncName is not None ):
    if ( RegStr.FuncName in [ "~/", "1/" ] ): # Those exact names: fractional/denominator with no supplementary Non-lin
      OutStr += RegStr.FuncName # no further processing needed
    elif ( ( len( RegStr.FuncName ) > 2 ) and ( RegStr.FuncName[ : 2 ] in [ "~/", "1/" ] ) ): # fractional/denominator non-lin
      OutStr += RegStr.FuncName[ : 2 ] + f"NonLinList[{ NonLinName2Idx[ RegStr.FuncName[ 2 : ] ] }].get_f()" # look-up without, but prepend
    else: # Non-fractional aka Numerator expression
      OutStr += f"NonLinList[{ NonLinName2Idx[ RegStr.FuncName ] }].get_f()"

  # -------------------------------- B. Subexpression handling
  OutStr += "("
  for i in range( len( RegStr.SubExpressions ) ):
    if ( RegStr.SubExpressions[ i ].VarName is not None ): # Not a constant
      TempSubExpr: Parser.SubExpression = copy.deepcopy( RegStr.SubExpressions[ i ] )

      if ( RegStr.SubExpressions[ i ].VarName == OutputVarName ):
        TempSubExpr.VarName = "OutVec" # reserve name to allow user to have y as variable
        is_AR = True
      else: TempSubExpr.VarName = f"Data[{ InputVarName2Idx[ TempSubExpr.VarName ] }]" # MA
      OutStr += f"({ TempSubExpr })"

    else: OutStr += f"({ RegStr.SubExpressions[ i ] })" # don't copy and process the variable if const or y

    if ( i != len( RegStr.Operators ) ): OutStr += " " + RegStr.Operators[ i ] + " " # one less oàperator than variables

  return ( OutStr + ")", is_AR )


############################################################################### Verify the Expression ##############################################################################
def VerifyParsedReg( Regressor: Parser.ParsedReg, NonLinNames: list[ str ], ModelVarNames: list[ str ], OutputVarName: str = 'y' ) -> None:
  '''Verifies that:
  - All used variables names are declared (contained into ModelVarNames)
  - All used non-linearity names are declared (contained into NonLinNames)
  - No non-negative lags of the Output Variable are used

  ### Inputs:
  - `Regressor`: (ParsedReg) containing the RegressorStr objects
  - `NonLinNames`: (list of str) containing the non-linearity names
  - `ModelVarNames`: (list of str) containing the input variable names
  - `OutputVarName`: (str) name of the output variable

  ### Raises:
  - `ValueError`: if a non-linearity name is not declared
  - `ValueError`: if a variable name is not declared
  - `ValueError`: if a non-negative lag for the output variable is found
  '''

  # Check if the needed non-linearity exist
  if ( Regressor.FuncName is not None ):
    CurrentName: str = Regressor.FuncName
    if ( ( CurrentName[ : 2 ] == '~/' ) or ( CurrentName[ : 2 ] == '1/' ) ): CurrentName = CurrentName[ 2 : ] # fractions are never stored as separate functions
    if ( ( CurrentName != '' ) and ( CurrentName not in NonLinNames ) ): # empty string if fraction/denominator term without non-lin
      raise ValueError( f"Non-linearity '{ CurrentName }' is not a declared Non-linearity" )

  for subexp in Regressor.SubExpressions:

    # Check if all needed variables exist
    if ( subexp.VarName is None ): continue # Skip constants

    if ( subexp.VarName not in ModelVarNames ): raise ValueError( f"Variable '{ subexp.VarName }' is not declared in ModelVarNames list[str]" )

    if ( subexp.VarName in OutputVarName ):
      if ( subexp.Lag is None ): raise ValueError( f"No lag for output variable '{ OutputVarName }', as found in the expression { subexp } is not supported (non-causality)" )
      if ( subexp.Lag is not None ):
        if ( subexp.Lag >= 0 ): raise ValueError( f"Positive lag for output variable '{ OutputVarName }' as found in expression { subexp } is not supported (non-causality)" )


############################################################################# Make Buffer Start System #############################################################################
def Make_BufferStartSystem( Expr: str ) -> str:
  '''Replaces variable references in expression with Toggle calls for buffer start.
  Converts Data[...] and OutVec[...] patterns to Toggle(...) function calls.

  ### Inputs:
  - `Expr`: (str) expression string containing Data and OutVec references

  ### Outputs:
  - (str) expression with Toggle function calls substituted
  '''

  Output = re.sub( r'Data\[(\d+)\]\[k(([-+])(\d+))?\]', # Regexp recognizing Data = Input regressors
                  lambda match: f"Toggle(0, { int( match.group( 1 ) ) }, k{ match.group( 2 ) if match.group( 2 ) is not None else '' }, Data)",
                  Expr
                )

  Output = re.sub( r'OutVec\[k(([-+])(\d+))?\]', # Regexp recognizing OutVec = Output regressors
                  lambda match: f"Toggle(1, None, k{ match.group( 1 ) if match.group( 1 ) is not None else '' }, None)",
                  Output # Overwrite
                )

  return ( Output )

################################################################################ Make System Lambdas ###############################################################################
def Make_SystemLambdas( ModelVarNames: list[ str ], OutputVarName: str, NonLinName2Idx: dict[ str, int ], InputVarName2Idx: dict[ str, int ], RegStrList: list[ Parser.ParsedReg ] ) -> tuple[ Optional[ Callable ], Optional[ Callable ], Callable, Callable ]:
  '''Creates lambda functions for system evaluation from parsed regressor expressions.
  Generates separate lambdas for MA numerator, MA denominator, buffer start system, and main system.

  ### Inputs:
  - `ModelVarNames`: (list of str) names of all model variables
  - `OutputVarName`: (str) name of the output variable
  - `NonLinName2Idx`: (dict) mapping from non-linearity name to index
  - `InputVarName2Idx`: (dict) mapping from input variable name to index in Data
  - `RegStrList`: (list of ParsedReg) parsed regressor expressions

  ### Outputs:
  - `SubSystem_MA_Num`: (callable or None) lambda for MA numerator terms
  - `SubSystem_MA_Den`: (callable or None) lambda for MA denominator terms
  - `System_BufferStart`: (callable) lambda for buffer start phase
  - `System_Main`: (callable) lambda for main system evaluation
  '''

  # ----------------------------------------------------------------------------- 0. Verify Regressors -----------------------------------------------------------------------------
  for reg in RegStrList: VerifyParsedReg( reg, NonLinName2Idx.keys(), ModelVarNames, OutputVarName ) # Are all required variables & nonLins passed + no sus lags?

  # --------------------------------------------------------------------------- 1. Create System strings ---------------------------------------------------------------------------
  # System subexpressions, both Den expressions remain '' for non-rational systems
  MA_NumExpr: str = ''; MA_DenExpr: str = ''; AR_NumExpr: str = ''; AR_DenExpr: str = ''

  for idx, reg in enumerate( RegStrList ):
    CurrentExpr, is_AR = ParsedReg2EvalStr( reg, InputVarName2Idx, NonLinName2Idx, OutputVarName )

    if ( reg.FuncName is None ): # Automatically a numerator term, as '~/' is considered a function name
      if ( is_AR ): AR_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "
      else:         MA_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "

    else: # Potentially denominator term
      if ( reg.FuncName[ : 2 ] == '~/' ):
        if ( is_AR ): AR_DenExpr += f"theta[{ idx }]*{ CurrentExpr.replace( '~/', '' ) } + "
        else:         MA_DenExpr += f"theta[{ idx }]*{ CurrentExpr.replace( '~/', '' ) } + "
      else:
        if ( is_AR ): AR_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "
        else:         MA_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "

  # remove the last ' + ' on all strings
  if ( MA_NumExpr != '' ): MA_NumExpr = MA_NumExpr[ : -3 ]
  if ( MA_DenExpr != '' ): MA_DenExpr = MA_DenExpr[ : -3 ]
  if ( AR_NumExpr != '' ): AR_NumExpr = AR_NumExpr[ : -3 ]
  if ( AR_DenExpr != '' ): AR_DenExpr = AR_DenExpr[ : -3 ]


  # Numerator Handling
  if ( ( AR_NumExpr == '' ) and ( MA_NumExpr == '' ) ): NumExpr = "1" # for denominator-only systems
  elif ( MA_NumExpr == '' ): NumExpr = AR_NumExpr
  elif ( AR_NumExpr == '' ): NumExpr = f"MA_Num[k]"
  else:                      NumExpr = f"MA_Num[k] + { AR_NumExpr }"

  isRational = ( MA_DenExpr != '' ) or ( AR_DenExpr != '' )

  # Optional Denominator Handling
  if ( isRational ): # Rational since at least one denom term exists
    if ( MA_DenExpr == '' ): DenExpr = f"1.0 + { AR_DenExpr }" # needs + 1 the way the arbo fits
    elif ( AR_DenExpr == '' ): DenExpr = f"1.0 + MA_Den[k]"
    else:                      DenExpr = f"1.0 + MA_Den[k] + { AR_DenExpr }"

  # --------------------------------------------------------------------------- 2. Create System lambdas ---------------------------------------------------------------------------

  # Bufferstart has no OutVec since accessed as class member
  Make_MA_System = lambda Expr: re.sub( r'Data\[(\d+)\]\[k(([-+])(\d+))?\]', # Regexp recognizing Data = Input regressors
                   lambda match: f"Toggle({ int( match.group( 1 ) ) },{ match.group( 2 ) if match.group( 2 ) is not None else '0' },Data)",
                   Expr )

  if ( MA_NumExpr != '' ):
    SubSystem_MA_Num = eval( f"lambda theta, Data, NonLinList, Toggle: { Make_MA_System( MA_NumExpr ) }" )
  else: SubSystem_MA_Num = None # if None, oscillate function checks just generates an array of zeros

  if ( MA_DenExpr != '' ):
    SubSystem_MA_Den = eval( f"lambda theta, Data, NonLinList, Toggle: { Make_MA_System( MA_DenExpr ) }" )
  else: SubSystem_MA_Den = None

  # The final system has the form y[k] = ( MA_Num[k] + AR_Sys_Num( Data, k ) ) / ( MA_Den[k] + AR_Sys_Denom( Data, k) )
  # Bufferstart has no OutVec since accessed as class member
  SystemStart = f"lambda k, theta, Data, NonLinList, MA_Num, MA_Den, " # Standard form even if som MA are None
  if ( isRational ):
    System_Main =        eval( SystemStart + f"OutVec: ({ NumExpr }) / ({ DenExpr })" )
    System_BufferStart = eval( SystemStart + f"Toggle: ({ Make_BufferStartSystem( NumExpr ) }) / ({ Make_BufferStartSystem( DenExpr ) })" )

  else: # Non-rational system
    System_Main =        eval( SystemStart + f"OutVec: { NumExpr }" )
    System_BufferStart = eval( SystemStart + f"Toggle: ({ Make_BufferStartSystem( NumExpr ) })" )

  return ( SubSystem_MA_Num, SubSystem_MA_Den, System_BufferStart, System_Main )


####################################################################################################################################################################################
#####                                                                           SYMBOLIC OSCILLATOR                                                                            #####
####################################################################################################################################################################################
class SymbolicOscillator:

  ###################################################################################### CTor ######################################################################################
  def __init__( self, ModelVarNames: list[ str ], NonLinearities: list[ NL.NonLinearity ], ExprList: list[ str ],
                theta: tor.Tensor, OutputVarName: str = 'y', dtype: tor.dtype = tor.get_default_dtype(), device: str = Device ) -> None:
    '''Generates the Regressor strings contained in the RegStr

    ### Inputs:
    - `ModelVarNames`: (list of str) containing the regressor names
    - `NonLinearities`: (list of NARMAX.NonLinearity) containing the non-linearity objects
    - `ExprList`: (list of str) containing the RegressorStr objects
    - `theta`: ((DsData.shape[1] + len( ExprList)),)-shaped torch tensor) containing the regression Coefficients
    - `OutputVarName`: (str) containing the name used to denote the output variable in the passed strings
    - `dtype`: (torch.dtype = torch.float64) containing the type of the output variable
    - `device`: (torch.device = rFOrLSr.device) containing the device of the output buffer
    '''
    # -------------------------------------------------------------------------------- Input checks --------------------------------------------------------------------------------
    if ( len( ModelVarNames ) == 0 ):  raise ValueError( "No Input variables were declared" )
    for i in range( len( ModelVarNames ) ):
      if ( not isinstance( ModelVarNames[ i ], str ) ): raise ValueError( f"ModelVarNames[{ i }] is not of type 'str'" )

    if ( len( ModelVarNames ) != len( set( ModelVarNames ) ) ): raise ValueError( "Duplicate names found in ModelVarNames" )

    if ( len( [ nl.get_Name() for nl in NonLinearities ] ) != len( set( [ nl.get_Name() for nl in NonLinearities ] ) ) ): raise ValueError( "Duplicate non-linearity names found" )

    if ( len( NonLinearities ) == 0 ): raise ValueError( "No Non-linearities were declared" )
    for i in range( len( NonLinearities ) ):
      if ( not isinstance( NonLinearities[ i ], NL.NonLinearity ) ): raise ValueError( f"NonLinearities[{ i }] is not of type 'NARMAX.NonLinearity'" )

    if ( len( ExprList ) == 0 ):     raise ValueError( "No Regressors were declared" )
    for i in range( len( ExprList ) ):
      if ( not isinstance( ExprList[ i ], str ) ): raise ValueError( f"RegStrList[{ i }] is not a string" )

    if ( not isinstance( theta, tor.Tensor ) ): raise ValueError( f"theta must be of type 'torch.Tensor'" )

    if ( not isinstance( OutputVarName, str ) ): raise ValueError( f"OutputVarName is not of type 'str'" )

    if ( not isinstance( dtype, tor.dtype ) ): raise ValueError( f"dtype is not of type 'torch.dtype'" )

    if ( theta.shape[ 0 ] != len( ExprList ) ): raise ValueError( f"theta length ({ theta.shape[ 0 ] }) must equal number of expressions ({ len( ExprList ) })" )

    # Just Poker that the passed device is valid, torch will certainly complain otherwise

    # -------------------------------------------------------------------------------- Data Storage --------------------------------------------------------------------------------
    # The Class stores only the data needed to oscillate, everything else is discarded

    # Processing parameters
    self.theta: tor.Tensor = theta.cpu()
    self.NonLinearities: list[ NL.NonLinearity ] = NonLinearities
    self.nExpressions: int = len( ExprList )
    self.nInputVars: int = len( ModelVarNames ) if ( OutputVarName not in ModelVarNames ) else len( ModelVarNames ) - 1 # Don't count the output variable, it has own buffer

    # Output Data parameters
    self.dtype: tor.dtype = dtype
    self.device: str = device # only needed for the output data, since internally the SymOsc uses the CPU

    # Systems Lambdas: all are Optional[ Callable ] of different but very long types
    self.SubSystem_MA_Num = None # Non-recursive Numerator part (GPU)
    self.SubSystem_MA_Den = None # Non-recursive Denominator part (GPU)
    self.System_BufferStart = None # Full system for buffer start, calls AR_Toggle to dispatch between storage and input (CPU)
    self.System_Main = None # Full system for main buffer (CPU)

    # MaxLags data for the current system. Uninitialized since strings not yet parsed
    self.MaxInputLag: int = 0 # Maximum negative delay for input data (x[k-j])
    self.MaxPositiveInputLag: int = 0 # To support acausal systems (in non-Output terms, since for Outputs VerifyExpression throws for lags >= 0)
    self.MaxOutputLag: int = 0 # Maximum negative delay for the output data (y[k-j])
    self.MaxStartLag: int = 0 # # The Lag of the furthest in the past reaching x or y term. # TODO: probably needs to be updated to support e[k] and multiple outputs

    # Internal storage for in/out past states to bridge buffer ends
    # Note: One can't use negative indices for the OutVec in python, since the user might use lags greater than the buffer or resize the buffer, which reinitializes it
    self.InputStorage: Optional[ tor.Tensor ] = None # 2D Tensor (nInputs x MaxNegLag): for any variable indexed by k-j with j > 0
    self.OutputStorage: Optional[ tor.Tensor ] = None # for OutVec[k-j] with j > 0

    self.OutVec: tor.Tensor = tor.zeros( 8, dtype = self.dtype, device = "cpu" ) # arbitrary length as placeholder: only known when user calls Oscillate()

    # --------------------------------------------------------------------- Convertion Dicts and Reg validation --------------------------------------------------------------------
    NonLinName2Idx: dict[ str, int ] = {} # key: name, value: index in NonLinearities
    for idx in range( len( NonLinearities ) ): NonLinName2Idx[ NonLinearities[ idx ].get_Name() ] = idx

    InputVarName2Idx: dict[ str, int ] = {} # key: name, value: index in Data
    for idx in range( len( ModelVarNames ) ):
      if ( ModelVarNames[ idx ] != OutputVarName ): InputVarName2Idx[ ModelVarNames[ idx ] ] = len( InputVarName2Idx ) # OutputVarName excluded from InputVarName2Idx, since InputVarName2Idx is for Data[] access

    OutVecError: str = "The name 'OutVec' is reserved for internal processing, plase rename that "
    if ( "OutVec" in InputVarName2Idx.keys() ): raise ValueError( OutVecError + "Variable" )
    if ( "OutVec" in NonLinName2Idx.keys() ):   raise ValueError( OutVecError + "NonLinearity" )

    RegStrList: list[ Parser.ParsedReg ] = [ Parser.ExpressionParser( expr ) for expr in ExprList ]

    # ---------------------------------------------------------------------------- Generate Expressions ----------------------------------------------------------------------------
    self.SubSystem_MA_Num, self.SubSystem_MA_Den, self.System_BufferStart, self.System_Main = Make_SystemLambdas( ModelVarNames, OutputVarName, NonLinName2Idx, InputVarName2Idx, RegStrList )

    # ----------------------------------------------------------------------- Find Maxlags and create buffers ----------------------------------------------------------------------
    for reg in RegStrList:
      for se in reg.SubExpressions:
        if ( ( se.Lag is None ) or ( se.Lag == 0 ) ): continue # Coeff which has no lag, also se.VarName is None

        if ( se.VarName == OutputVarName ): self.MaxOutputLag = max( self.MaxOutputLag, abs( se.Lag ) )
        else: # Input Data lags
          if ( se.Lag < 0 ): self.MaxInputLag = max( self.MaxInputLag, abs( se.Lag ) )
          elif ( se.Lag > 0 ): self.MaxPositiveInputLag = max( self.MaxPositiveInputLag, se.Lag )

    self.OutputStorage = tor.zeros( self.MaxOutputLag, dtype = self.dtype, device = "cpu" )
    self.InputStorage = tor.zeros( ( self.nInputVars, self.MaxInputLag ), dtype = self.dtype, device = "cpu" )
    self.MaxStartLag = max( self.MaxInputLag, self.MaxOutputLag ) # The Lag of the furthest in the past reaching x or y term. # TODO: probably needs to be updated to support e[k] and multiple outputs


  ################################################################################## theta setter ##################################################################################
  def set_theta( self, theta: tor.Tensor ) -> None:
    '''Setter for the regression coefficients.

    ### Input:
    - `theta`: ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    if ( not isinstance( theta, tor.Tensor ) ): raise ValueError( "theta must be a torch.Tensor" )
    if ( theta.shape[ 0 ] != self.nExpressions ): raise ValueError( f"theta has wrong dimension, expected { self.nExpressions }" )
    self.theta = theta.cpu()


  ################################################################################## theta getter ##################################################################################
  def get_theta( self ) -> tor.Tensor:
    '''Getter for the regression coefficients.

    ### Output:
    - ( ( nr, ) - sized float torch.tensor ) containing the estimated regression coefficients'''
    return ( self.theta.to( self.device ) )


  ############################################################################## Output Storage setter #############################################################################
  def set_OutputStorage( self, PreviousOutput: tor.Tensor ) -> None:
    '''Setter for the output storage. Allows to give the system information about its outputs generated before the to-be-passed data. (y[k-j])

    ### Input:
    - `PreviousOutput`: ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    if ( PreviousOutput.shape != self.OutputStorage.shape ): raise ValueError( f"PreviousOutput has wrong dimension, expected { self.OutputStorage.shape }" )
    if ( self.OutputStorage.dtype != PreviousOutput.dtype ): raise ValueError( f"PreviousOutput has wrong data type, expected { self.OutputStorage.dtype }" )
    self.OutputStorage = PreviousOutput.cpu()


  ############################################################################## Output Storage getter #############################################################################
  def get_OutputStorage( self ) -> tor.Tensor:
    '''Getter for the storage vector containing the required last buffer's past values required for system operation.

    ### Output:
    - ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    if ( self.OutputStorage is None ): raise ValueError( "Accessing empty OutputStorage" ) # Impossible since OutputStorage is initialized in __init__
    return ( self.OutputStorage.to( self.device ) )


  ############################################################################## Input Storage setter ##############################################################################
  def set_InputStorage( self, InputStorage: tor.Tensor ) -> None:
    '''Setter for the input storage. Allows to give the system information about its inputs before the to-be-passed data. (x[k-j])

    ### Input:
    - `InputStorage`: ( ( nr, ) - sized float torch.tensor ) containing the estimated regression coefficients'''

    if ( self.InputStorage is not None ): # Check valid dimensions only if existing buffer. Normally always the case since set by __init__
      if ( InputStorage.shape != self.InputStorage.shape ): raise ValueError( f"InputStorage has wrong dimension, expected { self.InputStorage.shape }" )
      if ( self.InputStorage.dtype != InputStorage.dtype ): raise ValueError( f"InputStorage has wrong data type, expected { self.InputStorage.dtype }" )
    self.InputStorage = InputStorage.cpu()


  ############################################################################## Input Storage getter ##############################################################################
  def get_InputStorage( self ) -> tor.Tensor:
    '''Getter for the input storage vector containing the required last buffer's past values required for system operation.

    ### Output:
    - ( ( nr, ) - sized float torch.tensor ) containing the estimated regression coefficients'''
    if ( self.InputStorage is None ): raise ValueError( "Accessing empty InputStorage" ) # Impossible since InputStorage is initialized in __init__
    return ( self.InputStorage.to( self.device ) )


  ################################################################################## flushBuffers ##################################################################################
  def zeroInternalStorage( self ) -> None:
    '''Zeros the internal buffers, such that the system isn't influenced by previous buffer's data'''
    # TODO: use the getters for the dimensions such that this also work in the None case
    self.InputStorage = tor.zeros( self.InputStorage.shape, dtype = self.dtype, device = "cpu" )
    self.OutputStorage = tor.zeros( self.OutputStorage.shape, dtype = self.dtype, device = "cpu" )


  ############################################################################ get number of regressors ############################################################################
  def get_nRegressors( self ) -> int:
    '''Returns the number of regressors (int): the number of different terms in the expression'''
    return ( self.nExpressions )


  ########################################################################## get number of input variables #########################################################################
  def get_nInputVars( self ) -> int:
    '''Returns the number of input variables (int): the number of different x[k-j]'''
    return ( self.nInputVars )


  ########################################################################### largest negative input lag ###########################################################################
  def get_MaxInputLag( self ) -> int:
    '''Returns the largest negative input lag (int): the largest j of all x[k-j]'''
    return ( self.MaxInputLag )


  ########################################################################### largest positive input lag ###########################################################################
  def get_MaxPositiveInputLag( self ) -> int:
    '''Returns the largest positive lag (int): the largest j of all x[k+j]'''
    return ( self.MaxPositiveInputLag )


  ########################################################################## largest negative output lag ###########################################################################
  def get_MaxOutputLag( self ) -> int:
    '''Returns the largest negative output lag (int): the largest j of all y[k-j]'''
    return ( self.MaxOutputLag )


  ################################################################################# Buffer Toggle ##################################################################################
  def Buffer_Toggle( self, VarNumber: int, k: int, Data: Sequence[ tor.Tensor ] ) -> tor.Tensor:
    '''Toggle function for block-wise buffer operations on input data.
    Handles negative lags by concatenating storage and input data.

    ### Inputs:
    - `VarNumber`: (int) index of the input variable
    - `k`: (int) lag index (negative for past values, zero for current)
    - `Data`: (sequence of torch.Tensor) input data tensors

    ### Returns:
    - (torch.Tensor) concatenated storage and data slice for the requested lag
    '''
    if ( k < 0 ):    return tor.concat( [ self.InputStorage[ VarNumber, k : ].to( Data[ VarNumber ].device ), # can't use get_InputStorage since it might move unnecessarily device
                                          Data[ VarNumber ][ : k ] ]
                                      )
    elif ( k == 0 ): return Data[ VarNumber ]
    else:            raise ValueError( "Lag must be negative or zero" )


  ################################################################################# Scalar Toggle ##################################################################################
  def Scalar_Toggle( self, DataOrOutVec: bool, VarNumber: int, k: int, Data: Sequence[ tor.Tensor ] ) -> tor.Tensor:
    ''' Helper function which toggles between the stored internal system state for k < 0 and incomming data for k >= 0.

    ### Inputs:
    - `DataOrOutVec`: (bool) 0 for Input data, 1 for Output (OutVec)
    - `VarNumber`: (int) index of the variable in the Data array
    - `k`: (int) sample index (negative for past values, non-negative for current/future)
    - `Data`: (sequence of torch.Tensor) input data tensors

    ### Returns:
    - (torch.Tensor) scalar value from either Data, InputStorage, OutVec, or OutputStorage

    ### Raises:
    - `ValueError`: if DataOrOutVec is neither 0 nor 1
    '''
    if ( DataOrOutVec == 0 ): # 0 for Input
      if ( k >= 0 ): return Data[ VarNumber ][ k ] # R[1/4] normal indexing since k >= 0
      else:          return self.InputStorage[ VarNumber, k ] # R[2/4] k is negative: indexing from the end

    elif ( DataOrOutVec == 1 ): # 1 for Output, same as above but without variable dispaching (currently only one output supported)
      if ( k >= 0 ): return self.OutVec[ k ] # R[3/4]
      else:          return self.OutputStorage[ k ] # R[4/4]

    else: raise ValueError( "Internal Error, please report this bug: 'DataOrOutVec is neither Data nor OutVec'" )


  #################################################################################### Oscillate ###################################################################################
  def Oscillate( self, Data: Sequence[ tor.Tensor ], theta: Optional[ tor.Tensor ] = None, DsData: Optional[ tor.Tensor ] = None ) -> tor.Tensor:
    ''' Function applying the NARMAX-system to the input data.
    DsData can is allowed to change the number of columns it has (if correctly mirrored in theta), since the system has no concept of it.

    ### Inputs:
    - `Data`: (sequence of 1D-torch.tensors having all the same length) containing the input variable vectors
    - `theta`: (optional (DsData.shape[1] + len( ExprList)),)-shaped torch.tensor) containing the regression Coefficients allowing to modulate theta
    - `DsData`: (optional 1D-tensor) containing any signal to be directly (no scaling, processing or storage) injected into the system
    '''
    # --------------------------------------------------------------------------- System Parameter Update --------------------------------------------------------------------------
    if ( not isinstance( Data, list | tuple ) ): raise ValueError( "Data must be a list or tuple" )
    if ( not isinstance( DsData, tor.Tensor | type( None ) ) ): raise ValueError( "DsData must be a torch.tensor or None" )
    if ( len( Data ) == 0 ): raise ValueError( "Data can't be empty, nothing to oscillate" )

    if ( DsData is not None ):
      if ( DsData.ndim != 1 ): raise ValueError( "DsData must be 1D" )
      if ( DsData.shape[ 0 ] != Data[ 0 ].shape[ 0 ] ): raise ValueError( f"DsData's dimension doesn't equal that of Data, being { Data[ 0 ].shape[ 0 ] }" )

    if ( theta is not None ): self.set_theta( theta )

    if ( self.OutVec.shape[ 0 ] != Data[ 0 ].shape[ 0 ] ): # stored to avoid re-allocating, update if necessary
      self.OutVec = tor.zeros( Data[ 0 ].shape[ 0 ], dtype = self.dtype, device = "cpu" ) # reinitialize since not used in state-storage

    if ( self.MaxInputLag + self.MaxPositiveInputLag > Data[ 0 ].shape[ 0 ] ):
      raise ValueError( f"Input buffer-size is smaller than the system's total lag range of { self.MaxInputLag + self.MaxPositiveInputLag }. Not illegal but not supported yet." )


    # --------------------------------------------------------------------------------- Processing ---------------------------------------------------------------------------------
    if ( DsData is not None ): self.OutVec = DsData.clone().cpu() # pre-allocate for performance
    else:                      self.OutVec = tor.zeros( Data[ 0 ].shape[ 0 ], dtype = self.dtype, device = "cpu" )

    # MA parts as block-operations (stay on user-given device, hopefully GPU)
    if ( self.SubSystem_MA_Num is not None ):
      MA_Num = self.SubSystem_MA_Num( self.theta.to( self.device, self.dtype ), Data, self.NonLinearities, self.Buffer_Toggle ).cpu()
    else: MA_Num = None

    if ( self.SubSystem_MA_Den is not None ):
      MA_Den = self.SubSystem_MA_Den( self.theta.to( self.device, self.dtype ), Data, self.NonLinearities, self.Buffer_Toggle ).cpu()
    else: MA_Den = None

    Data = [ x.cpu() for x in Data ] # force all to CPU

    for k in range( 0, self.MaxStartLag ): # Buffer start procedure, with dispatch to internal state via toggle
      self.OutVec[ k ] += self.System_BufferStart( k, self.theta, Data, self.NonLinearities, MA_Num, MA_Den, self.Scalar_Toggle )

    for k in range( self.MaxStartLag, Data[ 0 ].shape[ 0 ] - self.MaxPositiveInputLag ): # Fully swung-in state
      self.OutVec[ k ] += self.System_Main( k, self.theta, Data, self.NonLinearities, MA_Num, MA_Den, self.OutVec )

    # ---------------------------------------------------------------------------- Internal State Update ---------------------------------------------------------------------------
    if ( self.MaxInputLag > 0 ): # Don't trigger for systems being memoryless in the input terms (only x[k] terms)
      for input in range ( len( Data ) ): # Must iterate one by one since Data is an arbirtary iterable not necessarily a 2D Tensor
        if ( Data[ input ].ndim != 1 ): raise ValueError( f"Input Data must be 1D. The { input }-th input is not" )
        self.InputStorage[ input ] = Data[ input ][ -self.MaxInputLag : ].clone()

    if ( self.MaxOutputLag > 0 ): self.OutputStorage = self.OutVec[ -self.MaxOutputLag : ].clone() # Don't trigger for non-recursive systems

    return ( self.OutVec.to( self.device ) )
