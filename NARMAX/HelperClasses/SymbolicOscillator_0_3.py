import re
import copy
import torch as tor

# same folder
from . import NonLinearity as NL
from . import Parser_0_2 as Parser

def ScopeLimitHF(): # Here to clarify that HF is only used for the device selection
  from .. import HelperFuncs # Folder above
  return HelperFuncs.Set_Tensortype_And_Device()

Device = ScopeLimitHF() # Done here to avoid calling it twice (here and in the main __init__.py)

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
  isAR = False # Flag tracking if the current regressor contains the output varaible making it auto-regressive

  # --------------------------- A. Function name / constant coefficient handling
  OutStr = "" # Covers the case of RegStr.FuncName is None
  if ( RegStr.FuncName is not None ):
    if ( RegStr.FuncName in ["~/", "1/"] ): # Those exact names: fractional/denominator with no supplementary Non-lin
      OutStr += RegStr.FuncName # no further processing needed
    elif ( ( len( RegStr.FuncName ) > 2 ) and ( RegStr.FuncName[:2] in ["~/", "1/"] ) ): # fractional/denominator non-lin
      OutStr += RegStr.FuncName[:2] + f"NonLinList[{ NonLinName2Idx[RegStr.FuncName[2:]] }].get_f()" # look-up without, but prepend
    else: # Non-fractional aka Numerator expression
      OutStr += f"NonLinList[{ NonLinName2Idx[RegStr.FuncName] }].get_f()"

  # --------------------------- B. Subexpression handling
  OutStr += "("
  for i in range( len( RegStr.SubExpressions ) ):
    if ( RegStr.SubExpressions[i].VarName is not None ):
      TempSubExpr = copy.deepcopy( RegStr.SubExpressions[i] )

      if ( RegStr.SubExpressions[i].VarName == OutputVarName ):
        TempSubExpr.VarName = "OutVec" # reserve name to allow user to have y as variable
        isAR = True
      else: TempSubExpr.VarName = f"Data[{ VarName2Idx[TempSubExpr.VarName] }]" # MA
      OutStr += f"({ TempSubExpr })"
    
    else: OutStr += f"({ RegStr.SubExpressions[i] })" # don't copy and process the variable if const or y

    if ( i != len( RegStr.Operators ) ): OutStr += " " + RegStr.Operators[i] + " " # one less oàperator than variables

  return ( OutStr + ")", isAR )


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
    if ( ( CurrentName != '' ) and ( CurrentName not in NonLinNames ) ): # empty string if fraction/denominator term without non-lin
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
                  lambda match: f"Toggle(1, None, k{ match.group(1) if match.group(1) is not None else '' }, None)",
                  Output # Overwrite
                )
  
  return ( Output )


# ################################################ Make System Lambdas #################################################
def Make_SystemLambdas( InputVarNames, OutputVarName, NonLinName2Idx, VarName2Idx, RegStrList ):
  # ----------------------------------------------- 0. Verify Regressors -----------------------------------------------
  for reg in RegStrList:
    VerifyParsedReg( reg, NonLinName2Idx.keys(), InputVarNames, OutputVarName ) # Are all required variables & nonLins passed + no sus lags?

  # --------------------------------------------- 1. Create System strings ---------------------------------------------
  # System subexpressions, both Den expressions remain '' for non-rational systems
  MA_NumExpr = ''; MA_DenExpr = ''; AR_NumExpr = ''; AR_DenExpr = ''

  for idx, reg in enumerate( RegStrList ):
    CurrentExpr, isAR = ParsedReg2EvalStr( reg, VarName2Idx, NonLinName2Idx, OutputVarName )

    if ( reg.FuncName is None ): # Automatically a numerator term, as '~/' is considered a function name
      if ( isAR ): AR_NumExpr +=  f"theta[{ idx }]*{ CurrentExpr } + "
      else:        MA_NumExpr +=  f"theta[{ idx }]*{ CurrentExpr } + "
  
    else: # Potentially denominator term
      if ( reg.FuncName[:2] == '~/' ):
        if ( isAR ): AR_DenExpr += f"theta[{ idx }]*{ CurrentExpr.replace( '~/', '' ) } + "
        else:        MA_DenExpr += f"theta[{ idx }]*{ CurrentExpr.replace( '~/', '' ) } + "
      else:
        if ( isAR ): AR_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "
        else:        MA_NumExpr += f"theta[{ idx }]*{ CurrentExpr } + "
  
  # remove the last ' + ' on all strings
  if ( MA_NumExpr != ''): MA_NumExpr = MA_NumExpr[:-3]
  if ( MA_DenExpr != ''): MA_DenExpr = MA_DenExpr[:-3]
  if ( AR_NumExpr != ''): AR_NumExpr = AR_NumExpr[:-3]
  if ( AR_DenExpr != ''): AR_DenExpr = AR_DenExpr[:-3]


  # Numerator Handling
  if ( ( AR_NumExpr == '' ) and ( MA_NumExpr == '' ) ): NumExpr = "1" # for denominator-only systems
  elif ( MA_NumExpr == '' ): NumExpr = AR_NumExpr
  elif ( AR_NumExpr == '' ): NumExpr = f"MA_Num[k]"
  else:                      NumExpr = f"MA_Num[k] + { AR_NumExpr }"

  isRational = ( MA_DenExpr != '' ) or ( AR_DenExpr != '' )

  # Optional Denominator Handling
  if ( isRational ): # Rational since at least one denom term exists
    if   ( MA_DenExpr == '' ): DenExpr = f"1.0 + { AR_DenExpr }" # needs + 1 the way the arbo fits
    elif ( AR_DenExpr == '' ): DenExpr = f"1.0 + MA_Den[k]"
    else:                      DenExpr = f"1.0 + MA_Den[k] + { AR_DenExpr }"

  # --------------------------------------------- 2. Create System lambdas ---------------------------------------------

  # Bufferstart has no OutVec since accessed as class member
  Make_MA_System = lambda Expr: re.sub( r'Data\[(\d+)\]\[k(([-+])(\d+))?\]', # Regexp recognizing Data = Input regressors
                   lambda match: f"Toggle({ int( match.group(1) ) },{ match.group(2) if match.group(2) is not None else '0' },Data)",
                   Expr)
  
  if ( MA_NumExpr != ''  ):
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


########################################################################################################################
#####                                             SYMBOLIC OSCILLATOR                                              #####
########################################################################################################################
class SymbolicOscillator:

  # ####################################################### CTor #######################################################
  def __init__( self, InputVarNames, NonLinearities, ExprList, theta, OutputVarName = 'y', dtype = tor.float64, device = Device ):
    """Generates the Regressor strings contained in the RegStr
    
    ### Inputs:
    - `InputVarNames`: (list of str) containing the regressor names
    - `NonLinearities`: (list of NARMAX.NonLinearity) containing the non-linearity objects
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
      if ( not isinstance( NonLinearities[i], NL.NonLinearity ) ): raise ValueError( f"NonLinearities[{ i }] is not of type 'NARMAX.NonLinearity'" )

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
    self.theta = theta.cpu()
    self.NonLinearities = NonLinearities
    self.nExpressions = len( ExprList )
    self.nInputVars = len( InputVarNames ) if ( OutputVarName not in InputVarNames ) else len( InputVarNames ) - 1 # Don't count the output variable, it has own buffer

    # Output Data parameters
    self.dtype = dtype
    self.device = device # only needed for the output data, since internally teh SymOsc uses the CPU

    # Systems Lambdas
    self.SubSystem_MA_Num = None # Non-recursive Numerator part (GPU)
    self.SubSystem_MA_Den = None # Non-recursive Denominator part (GPU)
    self.System_BufferStart = None # Full system for buffer start, calls AR_Toggle to dispatch between storage and input (CPU)
    self.System_Main = None # Full system for main buffer (CPU)

    # MaxLags data for the current system. Uninitialized since strings not yet parsed
    self.MaxNegLag = 0 # Maximum negative delay for input data
    self.MaxPosLag = 0 # To support acausal systems (in non-Output terms, since for Outputs VerifyExpression throws for lags >= 0)
    self.MaxOutputLag = 0 # Maximum negative delay for the output data
    
    # Internal storage for in/out past states to bridge buffer ends
    # Note: one can't use negative indices for the OutVec in python, since the user might use lags greater than the buffer or resize the buffer, which reinitializes it
    self.InputStorage = None # for any variable indexed by k-j with j > 0
    self.OutputStorage = None # for OutVec[k-j] with j > 0
    
    self.OutVec = tor.zeros( 8, dtype = self.dtype, device = "cpu" ) # arbitrary length as placeholder: only known when user calls Oscillate()

    # --------------------------------------- Convertion Dicts and Reg validation --------------------------------------
    NonLinName2Idx = {} # key: name, value: index in NonLinearities
    for idx in range( len( NonLinearities ) ): NonLinName2Idx[ NonLinearities[idx].get_Name() ] = idx

    VarName2Idx = {} # key: name, value: index in Data
    for idx in range( len( InputVarNames ) ): VarName2Idx[ InputVarNames[idx] ] = idx
    
    OutVecError = "The name 'OutVec' is reserved for internal processing, plase rename that "
    if ( "OutVec" in VarName2Idx.keys() ):    raise ValueError( OutVecError + "Variable" )
    if ( "OutVec" in NonLinName2Idx.keys() ): raise ValueError( OutVecError + "NonLinearity" )

    RegStrList = [ Parser.ExpressionParser( expr ) for expr in ExprList ] # Parsed Regressor Objects

    # ---------------------------------------------- Generate Expressions ----------------------------------------------
    self.SubSystem_MA_Num, self.SubSystem_MA_Den, self.System_BufferStart, self.System_Main = \
      Make_SystemLambdas( InputVarNames, OutputVarName, NonLinName2Idx, VarName2Idx, RegStrList )

    # ----------------------------------------- Find Maxlags and create buffers ----------------------------------------
    for reg in RegStrList:
      for se in reg.SubExpressions:
        if ( ( se.Lag is None ) or ( se.Lag == 0 ) ): continue # Coeff which has no lag, also se.VarName is None

        if ( se.VarName == OutputVarName ): self.MaxOutputLag = max( self.MaxOutputLag, abs( se.Lag ) )    
        else: # Input Data lags
          if   ( se.Lag < 0 ): self.MaxNegLag = max( self.MaxNegLag, abs( se.Lag ) )
          elif ( se.Lag > 0 ): self.MaxPosLag = max( self.MaxPosLag, se.Lag )

    self.OutputStorage = tor.zeros( self.MaxOutputLag, dtype = self.dtype, device = "cpu" )
    self.InputStorage = tor.zeros( ( self.nInputVars, self.MaxNegLag ), dtype = self.dtype, device = "cpu" )


  # ################################################### theta setter ###################################################
  def set_theta( self, theta ):
    '''Setter for the regression coefficients.

    ### Input:
    - `theta`: ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    if ( theta.shape[0] != self.nExpressions ): raise ValueError( f"theta has wrong dimension, expected { self.nExpressions }" )
    self.theta = theta.cpu()
  

  # ################################################### theta getter ###################################################
  def get_theta( self ):
    '''Getter for the regression coefficients.

    ### Output:
    - ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients'''
    return ( self.theta.to( self.device ) )


  # ############################################### Output Storage setter ##############################################
  def set_OutputStorage( self, PreviousOutput ):
    '''Setter for the output storage. Allows to give the system information about its outputs generated before the to-be-passed data. (y[k-j])

    ### Input:
    - `PreviousOutput`: ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    if ( PreviousOutput.shape != self.OutputStorage.shape ): raise ValueError( f"PreviousOutput has wrong dimension, expected { self.OutputStorage.shape }" )
    if ( self.OutputStorage.dtype != PreviousOutput.dtype ): raise ValueError( f"PreviousOutput has wrong data type, expected { self.OutputStorage.dtype }" )
    self.OutputStorage = PreviousOutput.cpu()


  # ############################################### Output Storage getter ##############################################
  def get_OutputStorage( self ):
    '''Getter for the storage vector containing the required last buffer's past values required for system operation.

    ### Output:
    - ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients
    '''
    return ( self.OutputStorage.to( self.device ) )
  

  # ############################################### Input Storage setter ###############################################
  def set_InputStorage( self, InputStorage ):
    '''Setter for the input storage. Allows to give the system information about its inputs before the to-be-passed data. (x[k-j])

    ### Input:
    - `InputStorage`: ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients'''
    if ( InputStorage.shape != self.InputStorage.shape ): raise ValueError( f"InputStorage has wrong dimension, expected { self.InputStorage.shape }" )
    if ( self.InputStorage.dtype != InputStorage.dtype ): raise ValueError( f"InputStorage has wrong data type, expected { self.InputStorage.dtype }" )
    self.InputStorage = InputStorage.cpu()


  # ############################################### Input Storage getter ###############################################
  def get_InputStorage( self ):
    '''Getter for the input storage vector containing the required last buffer's past values required for system operation.

    ### Output:
    - ( (nr,)-sized float torch.tensor ) containing the estimated regression coefficients'''
    return ( self.InputStorage.to( self.device ) )


  # ################################################### flushBuffers ###################################################
  def zeroInternalStorage( self ):
    '''Zeros the internal buffers, such that the system isn't influenced by previous buffer's data'''
    self.InputStorage =  tor.zeros( self.InputStorage.shape,  dtype = self.dtype, device = "cpu" )
    self.OutputStorage = tor.zeros( self.OutputStorage.shape, dtype = self.dtype, device = "cpu" )


  # ############################################# get number of regressors #############################################
  def get_nRegressors( self ): 
    '''Returns the number of regressors (int)'''
    return ( self.nExpressions )
  

  # ########################################### get number of input variables ###########################################
  def get_nInputVars( self ):
    '''Returns the number of input variables (int)'''
    return ( self.nInputVars )
  

  # ############################################ largest negative input lag ############################################
  def get_MaxNegInputLag( self ):
    '''Returns the largest negative input lag (int)'''
    return ( self.MaxNegLag )


  # ############################################ largest positive input lag ############################################
  def get_MaxPosInputLag( self ):
    '''Returns the largest positive lag (int)'''
    return ( self.MaxPosLag )
  

  # ########################################### largest negative output lag ############################################
  def get_MaxNegOutputLag( self ):
    '''Returns the largest negative output lag (int)'''
    return ( self.MaxOutputLag )
  

  # ################################################## Buffer Toggle ###################################################
  def Buffer_Toggle( self, VarNumber, k, Data ):
    if ( k < 0 ): return tor.concat( [ self.InputStorage[ VarNumber, k: ].to( Data[ VarNumber ].device ), 
                                         Data[ VarNumber ][ :k ] ]
                                   )
    elif ( k == 0 ): return Data[ VarNumber ]
    else:         raise ValueError( "Lag must be negative or zero" )


  # ################################################## Scalar Toggle ###################################################
  def Scalar_Toggle( self, DataOrOutVec, VarNumber, k, Data ):
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
      self.OutVec = tor.zeros( Data[0].shape[0], dtype = self.dtype, device = "cpu" ) # reinitialize since not used in state-storage

    if ( self.MaxNegLag + self.MaxPosLag > Data[0].shape[0] ):
      raise ValueError( f"Input buffer-size is smaller than the system's total lag range of { self.MaxNegLag + self.MaxPosLag }. Not illegal but not supported yet." )

    # --------------------------------------------------- Processing ---------------------------------------------------
    if ( DsData is not None ): self.OutVec = DsData.clone().cpu() # pre-allocate for performance
    else:                      self.OutVec = tor.zeros( Data[0].shape, dtype = self.dtype, device = "cpu" )

    # MA parts as block-operations (stay on user-given device, hopefully GPU)
    if ( self.SubSystem_MA_Num is not None ):
      MA_Num = self.SubSystem_MA_Num( self.theta, Data, self.NonLinearities, self.Buffer_Toggle ).cpu()
    else: MA_Num = None

    if ( self.SubSystem_MA_Den is not None ):
      MA_Den = self.SubSystem_MA_Den( self.theta, Data, self.NonLinearities, self.Buffer_Toggle ).cpu()
    else: MA_Den = None

    Data = [x.cpu() for x in Data] # force all to CPU

    for k in range( 0, self.MaxNegLag ): # Buffer start procedure, with dispatch to internal state via toggle
      self.OutVec[k] += self.System_BufferStart( k, self.theta, Data, self.NonLinearities, MA_Num, MA_Den, self.Scalar_Toggle )

    for k in range( self.MaxNegLag, Data[0].shape[0] - self.MaxPosLag ): # Fully swung-in state
      self.OutVec[k] += self.System_Main( k, self.theta, Data, self.NonLinearities, MA_Num, MA_Den, self.OutVec)

    # --------------------------------------------- Internal State Update ----------------------------------------------
    if ( self.MaxNegLag > 0 ): # This doesn't trigger for systems being memoryless in the input terms (only x[k] terms)
      for input in range ( len( Data ) ): # Must iterate one by one since Data is an arbirtary iterable not necessarily a 2D Tensor
        if ( Data[input].ndim != 1 ): raise ValueError( f"Input Data must be 1D. The { input }-th input is not" )
        self.InputStorage[ input ] = Data[ input ][ -self.MaxNegLag : ].clone()
    
    if ( self.MaxOutputLag > 0 ): # This doesn't trigger for non-recursive systems
      self.OutputStorage = self.OutVec[ -self.MaxOutputLag : ].clone() # keep last outputs for next buffer
    
    return ( self.OutVec.to( self.device ) )


########################################################################################################################
#####                                                  UNIT TESTS                                                  #####
########################################################################################################################

if ( __name__ == '__main__' ):
  pass
   # TODO: unit test with some basic expression and all getters and some hardcodede result