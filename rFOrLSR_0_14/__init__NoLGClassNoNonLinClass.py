# ######################################################################################## Imports ###################################################################################
# Math
import numpy as np
import torch as tor
import matplotlib.pyplot as plt

# Rest of the Lib. "from ."" is same Folder and ".Name" is subfoldern import
from .HelperClasses import Queue # for the BFS Arborescent Regression
from .CTors import __init__ # import all constuctors and helper functions
from . import Morphing
from . import HelperFuncs as HF
# TODO note how the Lib.submodule is made to get a name but not load everything into the namespace, as from .CTors import * adds it to the library amespace
# TODO LookUp Dict class

import dill # memory dumps
import tqdm
import timeit

HF.Set_Tensortype_And_Device()


# ############################################################################ Variable Selection Procedure #############################################################################
def MaxLagPlotter( x, y, MaxLags = ( 30, 30 ), MaxOrder = 4, VarianceAcceptThreshold = 0.95 ):
  '''Variable selection function determining the maximum lags for y and x for rFOrLSR dictionary sparcification.
  This function is NARMAX specific since lagged variables are checked using arbitrary order NARX models rather than Taylor expansions.
  
  ### Inputs:
  -`x`: ( (p,)-shaped torch.tensor ) containing the system input
  -`y`: ( (p,)-shaped torch.ensor ) containing the system output signal
  -`MaxLags`: (2D int Tuple) containing the maximum lags for n_b and n_a, defaults to 30 for both ( n_b, n_a )
  -`MaxOrder`: (int > 0) Maximum approximation order used for the estimation
  -`VarianceAcceptThreshold`: ( float ) the minimum explained variance of the NARMAX expansion to estimate the needed delays
  
  ### Output:
  - `Modelorder`: (int) The chosen expansion's order, to be passed as (monomial) ExpansionOrder parameter to RegressionMatrix ( Dictionary CTor )
  - `Grid`: ( MaxLags[0], MaxLags[1] )-shaped np.array containing the ERR values displayed by the plot
  '''
  
  # Bullshit prevention:
  if ( ( x.ndim != 1 ) or ( y.ndim != 1 ) ):          raise AssertionError( "x or y is not a (p,)-shaped Tensor" )
  if ( len( x ) != len( y ) ):                        raise AssertionError( "x and y have not the same length" )
  if ( MaxOrder < 1 or not isinstance( MaxOrder, int ) ): raise AssertionError( "MaxOrder must be an int >= 1" )
  
  # --------------------------------------------------------------------------------- Ds only FOrLSR ------------------------------------------------------------------------------
  def MiniFOrLSR ( y, Ds ):
    ''' Very minimal Version of FOrLSR, as only the ERR of imposed terms is necessary '''
    y = tor.ravel( y ) # simplifies the code below
    s2y = (y @ y).item() # mean free observation empiric variance
    ERR = [] # list of Error reduction ratios of all selected regressors ( Dc and Ds )
    Psi = tor.empty( ( len( y ), 0 ) ); Psi_n = tor.empty( ( len( y ), 0 ) ) # Create empty ( p, 0 )-sized matrices to simplify the code below

    # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
    Psi = Ds[:, 0, None] # unnormed orthogonal regressor matrix ( already centered ) reshaped as column
    n_Omega = HF.Norm2( Psi ) # squared euclidean norm of Omega or fudge factor
    Psi_n = Psi / n_Omega # normed orthogonal regressor matrix
    ERR.append( ( ( Psi_n.T @ y ).item() )**2 * n_Omega / s2y ) # W[-1]^2 * n_Omega/s2y ) as usual but without storing W
    
    for col in range( 1, Ds.shape[1] ): # iterate over columns, start after position 1
      # Computations
      Omega = Ds[:, col] - Psi_n @ ( Psi.T @ Ds[:, col] ) # orthogonalize only the current column ( no reshape needed )
      n_Omega = HF.Norm2( Omega )  # squared euclidean norm of Omega or fudge factor
      ERR.append( ( ( Omega @ y ).item() / n_Omega )**2 * n_Omega / s2y ) # W[-1]^2 * n_Omega/s2y ) as usual but without storing W
      
      # Data storage, add current regressor
      Psi = tor.column_stack( ( Psi, Omega ) ) # unnormed matrix
      Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) ) # normed matrix
    
    return ( np.sum( ERR ) )

  # --------------------------------------------------------------------------------- Model order evaluation ------------------------------------------------------------------------------
  ModelOrder = 0 # try linear model first, incremented in the while to 1
  ModelExplainedVariance = 0 # Init to zero since not computed

  print( "Estimating model order." )
  while ( 5 ): # if less than Minvariance variance is explained, redo the analysis with a higher order model, since maxlag= max variance
    ModelOrder += 1 # increase order
    y_cut, RegMat = CTors.LagAndExpand( ( x, y ), MaxLags, ExpansionOrder = ModelOrder )[:2] # construct linear regressor matrix
    ModelExplainedVariance = MiniFOrLSR( y_cut - y_cut.mean(), RegMat - RegMat.mean( axis = 1 ) ) # pass y separately and exclude it in Ds, then sum of all ERRs
    
    if ( ( ModelOrder > MaxOrder + 1 ) or ( ModelExplainedVariance > VarianceAcceptThreshold ) ): break # do while condition, +1 since next iteration will exceed limit
    
  print( "\nA order", ModelOrder, "model explaining", 100 * ModelExplainedVariance, "% of the variance was selected. Computing the plot:" )
  
  # --------------------------------------------------------------------------------- Model computation ------------------------------------------------------------------------------
  Grid = np.full( ( MaxLags[1] + 1, MaxLags[0] + 1 ), np.nan ) # y's are rows and the x's columns to have a correct graph orientation, +1 due to x[k], y[k]
  ProgressBar = tqdm.tqdm( total = Grid.size ) # Initialise progressbar while declaring total number of iterations

  # Unintuitively, it's the MiniFOrLSR which accoutns for 99.99% of the time, not RegressorMatrix, so optimizing that out isn't of interest.
  for na in range( MaxLags[1] + 1 ): # iterate over y values, +1 to contain the value ( python indexing )
    for nb in range( MaxLags[0] + 1 ): # iterate over x values, +1 to contain the value ( python indexing )
      y_cut, RegMat = CTors.LagAndExpand( ( x, y ), ( nb, na ), ExpansionOrder = ModelOrder )[:2] # construct regressor matrix, must be reconstructed each time, due to cross terms
      Grid[na, nb] = MiniFOrLSR( y_cut - y_cut.mean(), RegMat - RegMat.mean( axis = 1 ) ) # store the ERR
      ProgressBar.update() # increase count
  ProgressBar.close() # Necessary
   
  # ---------------------------------------------------------------------------------- Plot -------------------------------------------------------------------------------------------------
  Y = np.arange( MaxLags[1] + 1 ); X = np.arange( MaxLags[0] + 1 ) # +1 to accomodate x[k] and y[k]

  Fig = plt.figure()
  Ax = plt.axes( projection = '3d' )
  Ax.plot_surface( *np.meshgrid( X, Y ), Grid, cmap = 'viridis', edgecolor = 'k', vmin = 0.9 * VarianceAcceptThreshold, vmax = 1 ) # vmin at 0.9 * VarianceAcceptThreshold, for more color range where its important
  Ax.set_ylabel( "y[k-i] terms" ); Ax.set_xlabel( "x[k-i] terms" )
  
  return ( ModelOrder, Grid )
  

  
# ################################################################################### Default Validation procedure ####################################################################################
def DefaultValidation( theta, L, ERR, V, MorphDict, DcFilterIdx = None ):
  '''
  Default Validation function based on time domain MAE.
  
  ### Input
  - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
  - `L`: ( (nr,)-sized int nd-array ) containing the regressors indices
  - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR ( not used but placeholder to adhere to AOrLSR standard )
  
  - `V`: ( dict ) containing the validation data to be passed to Dc's Ctors:
    → `V[Data]`: ( list of 2D-tuples of float nd-arrays ) containing x and y
    → `V[DsData]`: ( list of float matrices or None ) containing all imposed terms column wise
    → `V[MaxLags]`: ( 2D tuple of float ) containing the respective maximum delays for x then y terms
    → `V[ExpansionOrder]`: ( int ) containing the monomial expansion's maximal summed power
    → `V[fPtrs]`: ( list of pointer to functions ) containing the regression's transformations to be passed to RegressorTransform
    → `V[fStrs]`: ( list of str ) containing the regression's transformations names to be passed to RegressorTransform
  
  - `MorphDict`: # ( dict ) # TODO

  - `MorphData`: ( list ) containing the requires information to recreate the morphed regressors
    → `index`: ( int ) containing the column number in Dc of the current regressor
    → `fs`: ( int ) containing the non-linearity's number
    → `LM`: ( list of int ) containing the indexes constituting the non-linearity's arguments
    → `ksi`: ( ( r, )-shaped Tensor ) containing the arguments linear combination's coefficients
  
  ### Output
  -`Error`: ( float ) containing the passed model's validation error on the validation set
  '''
  
  # --------------------------------------------------------------------  bullshit prevention with readable errors for the user -----------------------------------------------------------------------
  if ( not isinstance( V, dict ) ): raise AssertionError( "The passed V datastructure is not a dictionary as expected" )
  
  for var in ["Data", "DsData", "MaxLags", "ExpansionOrder", "fPtrs", "fStrs", "nC", "MakeRational"]:
    if ( var not in V.keys() ): raise AssertionError( f"The validation datastructure contains no '{ var }' entry" )
  
  if ( not isinstance( V["Data"], list ) ):  raise AssertionError( "V's 'Data' entry is expected to be a list" )
  if ( not isinstance( V["fPtrs"], list ) ): raise AssertionError( "V's 'fPtrs' entry is expected to be a list" )
  if ( not isinstance( V["fStrs"], list ) ): raise AssertionError( "V's 'fStrs' entry is expected to be a list" )
  if ( not isinstance( V["nC"], int ) ):     raise AssertionError( "V's 'nC' entry is expected to be an int" )
  
  # --------------------------------------------------------------------  Validation computation loop -----------------------------------------------------------------------
  Error = 0 # total relative error
  qx = V["MaxLags"][0]; qy = V["MaxLags"][1]; q = max( ( qx, qy ) ) # all swung in states ( x, y, max( x,y ) )
  
  for Sig in range( len( V["Data"] ) ): # iterate over all passed Data tuples
    y, RegMat, RegNames = CTors.Lagger( Data = ( V["Data"][Sig][0][q:], V["Data"][Sig][1][q:] ), MaxLags = ( qx, qy ) ) # Create the delayed signal terms
    RegMat, RegNames = CTors.Expander( RegMat, RegNames, ExpansionOrder = V["ExpansionOrder"] ) # Monomial expand the regressors
    RegMat, RegNames = CTors.NonLinearizer( y, RegMat, RegNames, V["fPtrs"], V["fStrs"], V["MakeRational"] ) # add the listed terms to the Regression matrix

    # Filter out same regressors as for the regression
    RegMat = RegMat[:, DcFilterIdx]
    RegNames = RegNames[DcFilterIdx]

    # Centering
    y -= tor.mean( y )
    Means = tor.mean( RegMat, axis = 0, keepdims = True ) # Store the means ( needed by the Morpher )
    RegMat -= Means # Center the regressors by subtracting their respecitve ( = columnwise ) means

    nC = RegMat.shape[1] # No added regressors to dict. nC needed to detect if morphed term or not

    yHat = tor.zeros( RegMat.shape[0] ) # the p of the validation

    if ( ( V["DsData"] is not None ) and ( V["DsData"] != [] ) ):
      yHat = V["DsData"][Sig] @ theta[:V["DsData"][Sig].shape[1]] # becomes a ( p,1 )-shaped Tensor
      nS = V["DsData"][Sig].shape[1] # number of cols in Validation Ds
    else: nS = 0

    for reg in range( len( L ) ): # iterate over all L ans check if morphed
      # TODO adapt to Pytorch
      if ( L[reg] >= nC ): # morphed regressor since index higher ( or equal since 0 based ) than original dictionary size
        nMorphed = L[reg] - nC # number of the morphed term in the chronological list of creation
        LM = MorphDict["MorphData"][nMorphed][2] # alias for clarity
        f = MorphDict["fPtrs"][ MorphDict["MorphData"][nMorphed][1] ] # alias for clarity
        Xl = RegMat[:, LM] + Means[:, LM]
        fT = f( Xl @ MorphDict["MorphData"][nMorphed][-1] ); fT -= tor.mean( fT ) # MorphData[reg][-1]
        yHat += theta[ nS + reg ] * fT

      else: yHat += theta[ nS + reg ] * RegMat[ :, L[reg] ] # normal regressor
    
    
    Error += tor.mean( tor.abs( y - yHat ) / tor.mean( tor.abs( y ) ) ) # relative MAE
    
  return ( Error / len( V["Data"] ) ) # norm by the number of signals ( not really necessary for AOrLSR but printed for the user )


# ######################################################################################################## ADMOrLSR Class ###############################################################################################
# **************************************************************************************************** Helper functions *****************************************************************************************
# --------------------------------------------------------------------------------------------------- All Combinations ------------------------------------------------------------------------------------------
def AllCombinations( ImposedRegs, InputSeq, INT_TYPE ): 
  '''Helper function creating all combiantions of imposed Regressors to construct the current node's children. Assumes that InputSeq has been flattened'''
  InputSeq = np.setdiff1d( InputSeq, ImposedRegs, True ) # make sure InputSeq doesn't contain the imposed terms (only needed for OOIT-predicted matches)
  # TODO this works but verify that OOIT predicted matches have the property that the imposed terms aren't necessary in the beginning

  Out = np.empty( ( len( InputSeq ), len( ImposedRegs ) + 1 ), dtype = INT_TYPE ) # dimensions are ( nNewRegressors=nCombinations, Combination lengh )
  Out[:, :-1] = ImposedRegs # all rows start with the imposed terms
  Out[:, -1] = InputSeq # the last columns are the new regressors
  return ( Out )


# ------------------------------------------------------------------------------------------------------ Same Start AOrLSR ---------------------------------------------------------------------------------------------
def SameStart( Item, Dict ):
  '''Helper function checking if Item matches the start of any element contained in LG. This is the check performed before the node creation.
  ### Input:
  - `Item`: 1D int array
  - `LG`: Dict where the keys are tuples of ints and the values are ints corresponding to where the keys are in LG.

  ### Output:
  - `Out`: If a corresponding LG element is found, return it (int) else an empty list ([])
  '''
  key = tuple( np.sort( Item ) )
  if ( key in Dict ): return ( Dict[key] ) # return index in LG
  else: return ( [] ) # no matching term was found during the iteration

# ------------------------------------------------------------------------------------------------------ FindMinInt ---------------------------------------------------------------------------------------------
def FindMinInt( nCols ):
  '''Function determining the numpy integer dtype necessary to hold the current nCols to reduce memory usage'''
  if   ( nCols <= np.iinfo( np.uint16 ).max ): return ( np.uint16 )
  elif ( nCols <= np.iinfo( np.uint32 ).max ): return ( np.uint32 ) # certainly sufficient in most cases
  else: return ( np.uint64 )



# ####################################################################################################### AOrLSR Class ############################################################################################
class ADMOrLSR:

  # ******************************************************************************************************** Init *************************************************************************************************
  def __init__( self, y = None, Ds = None, DsNames = None, Dc = None, DcNames = None, tol1 = 0.001, tol2 = 0.001, MaxDepth = 5, F = None, V = None, verbose = False, MorphDict = None, FileName = None, SaveFrequency = 0 ):
    '''
    This can be called without arguments if the object is supposed to be filled by the load() function.

    ### Input:
    - `y`: ( (p,)-torch.Tensor ) System response
    - `Ds`: (None or (p, nS )-torch.Tensor ) Selected regressor matrix where each column corresponds to a (pre-) selected regressor
    - `DsNames`: (None or (nS,)-np.array of strings) Names of user pre-selected regressors
    - `Dc`: (None or (p, nC)-torch.Tensor ) Candidate regressor matrix stacking candidate regressors column-wise
    - `DcNames`: (None or (nC,)-np.array of strings) Names of candidate regressor matrices (only used by PlotAndPrint function)
    - 'tol1': (float) containing the maximum summed ERR threshold used for the root regression
    - 'tol2': (float) containing the maximum summed ERR threshold used for all non-root nodes
    - `MaxDepth`: (float) containing the maximum number of levels to be iterated
    - `F`: (pointer to function) linking to the custom validation scalar function designed to work with the dict V
    - `V`: (dict) containing whatever data F needs to output the validation metric float
    - `verbose`: (bool) containing whether to print the current state of the rFOrLSR (only meaningful for regressions with many terms)
    - TODO morphing Dict description and save variables description
    - `FileName`: (str) containing the path and file name to save the backups into
    - `SaveFrequency`: (float) containing the frequency in seconds at which to save the backups (10 = every 10 minutes)

    '''
    # ------------------------------------------------------------------------------------------ Passed Arguments ---------------------------------------------------------------------------------
    # Copy everything into the class

    if ( y is not None ): # Only when initializing empty arbo when loading from a file
      self.y = tor.ravel( y - tor.mean( y ) ) # flatski to (p,) vector for rFOrLSR
      if ( tor.isnan( tor.sum( self.y ) ) ): raise AssertionError( "Your system output y somehow yields NaNs, Bruh" )
    else: self.y = None

    self.Ds = Ds
    self.Dc = Dc # Center the regressors by subtracting their respecitve ( = columnwise ) means
    self.DcMeans = None # Overwritten below, when Dc's non-redundant size (nC) is known after filtering duplicate regressors
    self.DcNames = DcNames
    self.DsNames = DsNames
    self.tol1 = tol1
    self.tol2 = tol2
    self.MaxDepth = MaxDepth
    self.F = F
    self.V = V
    self.verbose = verbose
    self.MorphDict = MorphDict
    self.FileName = FileName
    self.SaveFrequency = SaveFrequency * 60 # transform into minutes
    self.INT_TYPE = np.int64 # default, if no DC present to overwrite it (thus not of any use since it only affects the arbo search)
    self.DcFilterIdx = None # indexset of all non-duplicate regressors in Dc

    # ------------------------------------------------------------------------------------------ Argument Processing ---------------------------------------------------------------------------------

    # ~~~~~~~~~~~~ DC Stuff
    if ( self.DcNames is not None ): # needs to be checked first since used in Dc processing -if below
      if ( type( self.DcNames ) is not np.ndarray ):  raise TypeError( "DcNames must be None or a numpy.ndarray" )
      if ( len( self.DcNames ) != self.Dc.shape[1] ): raise TypeError( "DcNames must be None or a numpy.ndarray of length nC" )

    if ( self.Dc is not None ):
        if ( type( self.Dc ) is not tor.Tensor ): raise TypeError( "Dc must be None or a torch.Tensor" )
        if ( tor.isnan( tor.sum( self.Dc ) ) ):   raise AssertionError( "Regressor Matrix Dc contains NaNs" )

        self.Dc, self.DcNames, self.DcFilterIdx = HF.RemoveDuplicates( self.Dc, self.DcNames )
        self.DcMeans = tor.mean( self.Dc, axis = 0, keepdims = True ) # Store the means ( needed by the Morpher )
        self.Dc -= self.DcMeans # Columnwise centering
        self.nC = self.Dc.shape[1] # store the updated number of regressors before Morphing
        self.INT_TYPE = FindMinInt( self.Dc.shape[1] ) # find size the index must at least be to account for all columns

    else: self.nC = 0 # no candidate regs if no Dc passed


    # ~~~~~~~~~~~~ Ds Stuff
    if ( ( self.Ds is not None ) and ( type( self.Ds ) is not tor.Tensor ) ): raise TypeError( "Ds must be None or a torch.Tensor" )

    if ( self.Ds is None ):
      if ( self.y is not None ): self.Ds = tor.empty( ( len( y ), 0 ) ) # create empty matrix, avoids some if/elses
      else: self.Ds = tor.empty( ( 0, 0 ) ) # No information on shape available, will be overwritten by load
      self.DsNames = np.empty( ( 0, ) )  # simplifies the code
    
    else: # A Ds is passed
      if ( tor.isnan( tor.sum( self.Ds ) ) ):         raise AssertionError( "Your Regressor Matrix Dc somehow yields NaNs, Bruh" )
      if ( type( self.DsNames ) is not np.ndarray ):  raise TypeError( "DsNames must be None or a numpy.ndarray" )
      if ( len( self.DsNames ) != self.Ds.shape[1] ): raise TypeError( "DsNames has not the same number of elements (columns) as Ds" )
      self.Ds -= tor.mean( Ds, axis = 0, keepdims = True ) # Columnwise centering
    self.nS = self.Ds.shape[1] # number of columns ( zero if above if is true )

    # ~~~~~~~~~~~~ Other Stuff
    if ( self.F is None ): self.F = lambda theta, L, ERR, Validation, MorphDic, : 1 - tor.sum( tor.tensor( ERR ) ) # default to explained variance if no validation function is passed

  
    # ------------------------------------------------------------------------------------------------ Internal Data ---------------------------------------------------------------------------------
    # the following variables are only used if Dc is not None
    # TODO Only create that U is None is passed by the user when supported by AOrLSR
    self.U = [ j for j in range( self.nC ) ] # unsused index set (uses nC instead of len( Dc ), since Dc might already be morphed )

    self.Q = Queue.Queue()
    self.LG = [] # Array to contain the actual regressions indices
    self.LookUpDict = {} # Hashtable used to quickly find if a regression is already computed

    self.Abort = False # Toggle activated at MaxDepth to finish regressions early

    self.MinLen = [] # overwritten by first iteration
    self.nTermsNext = 0 # Nodes in next level, overwritten by first iteration
    self.nTermsCurrent = 0 # Nodes in curretn level, overwritten by first iteration
    self.nComputed = 0 # Nodes computed in current level, stores the current tqdm state

    # For stats only (not needed by the arbo):
    self.TotalNodes = 1 # Nodes in the arbo ( starts at 1 since root counts )
    self.NotSkipped = 0 # Actually computed regressions ( =0 is used to detect if fit is called for the first time or not )
    self.AbortedRegs = 0 # Not OOIT - computed regressions, since longer than shortest known

    # ------------------------------------------------------------------------------------------------ Results ---------------------------------------------------------------------------------
    # Data of the regressor selected by the validation
    self.theta = None # theta is None is used as verification if the validation function has been done, so no-touchy
    self.ERR = None
    self.L = None



  # *********************************************************************************************** Memory dump **********************************************************************************
  def MemoryDump( self, nComputed ):
    '''
    Dumps the current Arborescence into a pickle file to load it later with load().
    '''
    ProgressCount = tqdm.tqdm( desc = "Creating backup", leave = False, unit = " Datastructures" )

    with open( self.FileName, 'wb' ) as file: # writes to or overwrites any existing file
    
      # User Inputs:
      dill.dump( self.y, file );              ProgressCount.update()
      dill.dump( self.Ds, file );             ProgressCount.update()
      dill.dump( self.DcMeans, file );        ProgressCount.update()
      dill.dump( self.Dc, file );             ProgressCount.update()
      dill.dump( self.DcNames, file );        ProgressCount.update()
      dill.dump( self.DsNames, file );        ProgressCount.update()
      dill.dump( self.tol1, file );           ProgressCount.update()
      dill.dump( self.tol2, file );           ProgressCount.update()
      dill.dump( self.MaxDepth, file );       ProgressCount.update()
      dill.dump( self.F, file );              ProgressCount.update()
      dill.dump( self.V, file );              ProgressCount.update()
      dill.dump( self.verbose, file );        ProgressCount.update()
      dill.dump( self.MorphDict, file );      ProgressCount.update()
      dill.dump( self.FileName, file );       ProgressCount.update()
      dill.dump( self.SaveFrequency, file );  ProgressCount.update()
      dill.dump( self.INT_TYPE, file );       ProgressCount.update()

      # Processing data
      dill.dump( self.nC, file );             ProgressCount.update()
      dill.dump( self.nS, file );             ProgressCount.update()
      dill.dump( self.U, file );              ProgressCount.update()
      dill.dump( self.Q, file );              ProgressCount.update()
      dill.dump( self.LG, file );             ProgressCount.update()
      dill.dump( self.LookUpDict, file );     ProgressCount.update()
      dill.dump( self.Abort, file );          ProgressCount.update()

      # Arbo Statistics
      dill.dump( self.MinLen, file );         ProgressCount.update()
      dill.dump( self.nTermsNext, file );     ProgressCount.update()
      dill.dump( self.nTermsCurrent, file );  ProgressCount.update()
      dill.dump( nComputed, file );           ProgressCount.update() # from function input, used to display the correct progress
      dill.dump( self.TotalNodes, file );     ProgressCount.update()
      dill.dump( self.NotSkipped, file );     ProgressCount.update()
      dill.dump( self.AbortedRegs, file );    ProgressCount.update()
    
    ProgressCount.close()


  # *********************************************************************************************** Memory Load **********************************************************************************
  def load( self, FileName, Print = True ): 
    ''' Loads the Arbo from a file pickled by the dill module into the current object to continue fitting
    
    ### Inputs:
    - `FileName`: ( str ) FileName to the pickle file
    '''

    with open( FileName, 'rb' ) as file:

      # User Inputs:
      self.y = dill.load( file )
      self.Ds = dill.load( file )
      self.DcMeans = dill.load( file )
      self.Dc = dill.load( file )
      self.DcNames = dill.load( file )
      self.DsNames = dill.load( file )
      self.tol1 = dill.load( file )
      self.tol2 = dill.load( file )
      self.MaxDepth = dill.load( file )
      self.F = dill.load( file )
      self.V = dill.load( file )
      self.verbose = dill.load( file )
      self.MorphDict = dill.load( file )
      self.FileName = dill.load( file )
      self.SaveFrequency = dill.load( file )
      self.INT_TYPE = dill.load( file )

      # Processing Data
      self.nC = dill.load( file )
      self.nS = dill.load( file )
      self.U = dill.load( file )
      self.Q = dill.load( file )
      self.LG = dill.load( file )
      self.LookUpDict = dill.load( file )
      self.Abort = dill.load( file )

      # Arbo Statistics
      self.MinLen = dill.load( file )
      self.nTermsNext = dill.load( file )
      self.nTermsCurrent = dill.load( file )
      self.nComputed = dill.load( file )
      self.TotalNodes = dill.load( file )
      self.NotSkipped = dill.load( file )
      self.AbortedRegs = dill.load( file )

      if ( Print ): # print some stats
        print( "\nTolerance: " + str(self.tol2), "\nArboDepth:", self.MaxDepth, "\nCurrent Dict-shape:", self.Dc.shape,
              "\nTraversed Nodes:", self.TotalNodes, "\nNot Skipped:", self.NotSkipped, "\nAborted Regs:", self.AbortedRegs )
        print("\nShortest Sequence at each level:")
        for i in range( len( self.Q.peek() ) ): print( f"Level { i }: { self.MinLen[i] }" ) 
        print() # print empty line

  # ************************************************************************************** rFOrLSR **************************************************************************************
  def rFOrLSR( self, y, Ds = None, Dc = None, U = None, tol = 0.001, MaxTerms = tor.inf, OutputAll = False, LI = None ): # Morphing Data
    '''
    Recursive Forward Orthogonal Regression function with Imposed terms capabilities expecting the regressors to be computed in advance and arranged in a regressor matrix.
    The Ds ( pre-selected dictionary ) terms ( if present ) are automatically accepted and orthogonalizd in appearance order, then Term selection is performed on Dc
    If the Regressor matrix Dc is None then the alorithm just uses Ds.
    The function assumes that the regressors and y[k] are centered vectors ( assured by the CTors and Transformers ), else the results will be relatively poor
    
    ### Inputs: (Most of them taken directly from the class)
    - `y`: ( (p,)-Tensor ) Observed Signal ( assumed mean-free )
    - `Ds`: ( None or ( p,nS )-Tensor ) Selected regressor matrix where each column corresponds to a ( pre- ) selected regressor. None is equivalent to an empty matrix
    - `Dc`: ( None or p,nC )-Tensor ) Candidate regressor matrix where each column corresponds to a candidate regressor
    - `U`: ( None or list of ints ) containing the column numbers of Dc the rFOrLSR is allowed to use, None defaults to all
    - `tol`: ( float ) maximum summed ERR threshold
    - `MaxTerms`: ( float ) determining the maximum number of terms to be contained in the regression before aborting
    - `OutputAll`: ( bool ) determining if Every Data structure is to be output
    # TODO LI

    Data taken from the Class:
    - `MDict`: ( Dictionary or None ) containing all the required morphing data or None if no morphing is desired
    - `Abort`: ( bool ) if the regression should be aborted if the number MaxTerms is exceeded
    - `verbose`: ( bool ) determining if a dot is printed when a regressor is added to the regression to notify advancement ( no bar since n of terms is unknown )
    → Dc can't be taken from the class since that woulf take away the possibility for validate to call with Dc = None
    → Ds also since it is modified by the Arbo
    
    ### Outputs: # TODO the number of outputs depends on what made the regression exit
    - `theta`: ( ( nr, )-array of float ) containing the regression coefficients, empty array if regression aborted
    - `L`: ( ( nr, )array of self.INT_TYPE ) containing the used terms' indices (DOESN'T CONTAIN Ds and LI )
    - `ERR`: ( ( nr, )-array of float ) containing the Error reduction ratios of all selected regressors ( Ds then Dc ), empty array if regression aborted
    '''

    # ------------------------------------------------------------------------------------ 0. Init --------------------------------------------------------------------------------
    MatSize = 0 # Solution matrix shape
    # all declared here since part of the output
    if ( self.verbose ): ProgressCount = tqdm.tqdm( desc = "Current Regression", leave = False, unit = " Regressors" ) # define measuring progress bar
    
    if ( Dc is not None ): # a dictionary serach must be performed ( not only imposed terms )
      if ( U is None ): U = [j for j in range( Dc.shape[1] )] # set of unused Dc indices ( elements in the set Dc )
      MatSize += len( U ) # can't use Dc.shape since not all terms might be allowed ( due to Arbo/morphing )
    
    if ( isinstance( Ds, tor.Tensor ) and Ds.shape == ( len( y ), 0 ) ): Ds = None # overwrite empty array as None
    if ( Ds is not None ):
      if ( Ds.shape == ( len( y ), ) ): Ds = Ds.reshape(-1, 1) # transform to single column
      MatSize += Ds.shape[1] # there are imposed terms
      if ( MaxTerms < Ds.shape[1] ): raise ValueError( f"MaxTerms ({ MaxTerms }) is smaller than imposed terms Ds.shape[1] ({ Ds.shape[1] })")

    if ( MatSize == 0 ): raise AssertionError( "Dc and Ds are None, thus no Regressors were passed for the algorithm to process" )
    
    L = [] # set of used regressor indices, needed for the output, even if Dc is None
    A = [] # Custom sparse martix for unitriangular matrices, storing only the upper diagonal elements in a list
    s2y = (y @ y).item() # mean free observation empiric variance
    ERR = [] # list of Error reduction ratios of all selected regressors ( Dc and Ds )
    s = 1 # iteration/column count ( 1 since zero-based but only used at 1nd iteration = loop )
    W = [] # Orthogonalized regressors coefficients
    Psi = tor.empty( ( len( y ), 0 ) ); Psi_n = tor.empty( ( len( y ), 0 ) ) # Create empty ( p,0 )-sized matrices to simplify the code below
    
    MOut = None # for the verification in case it's not overwritten by the morphing
    #  ----------------------------------------------------------------------- 1. Imposed terms orthogonalization ------------------------------------------------------------------------
    
    if ( Ds is not None ): # A matrix was passed as Ds, thus there are pre-selected regressors which are taken in order
      # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
      Psi = Ds[:, 0, None] # unnormed orthogonal regressor matrix ( already centered ) reshaped as column
      n_Omega = HF.Norm2( Psi ) # squared euclidean norm of Omega or fudge factor
      Psi_n = Psi / n_Omega # normed orthogonal regressor matrix
      W.append( ( Psi_n.T @ y ).item() ) # store orthogonal regression coefficient ( w ), extract value from arrray with [0]
      ERR.append( W[-1]**2 * n_Omega / s2y )
      if ( self.verbose ): ProgressCount.update() # Update counter
      
      for col in range( 1, Ds.shape[1] ): # iterate over columns, start after position 1
        # Computations
        Omega = Ds[:, col] - Psi_n @ ( Psi.T @ Ds[:, col] ) # orthogonalize only the current column ( no reshape needed )
        n_Omega = HF.Norm2( Omega )  # squared euclidean norm of Omega or fudge factor
        W.append( ( Omega @ y ).item() / n_Omega ) # store orthogonal regression coefficient ( w )
        ERR.append( W[-1]**2 * n_Omega / s2y ) # store this regressors ERR, eliminating the y[k-i] terms is done in the return
        
        # Data storage, add current regressor
        A.append( Psi_n.T @ Ds[:, col, None] ) # Multiply all normed orthogonalized regressors with the currently chosen unorthogonalized regressor
        Psi = tor.column_stack( ( Psi, Omega ) ) # unnormed matrix
        Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) ) # normed matrix
        s += 1 # increment the A-matrix column count
        if ( self.verbose ): ProgressCount.update() # Update counter

    # Term selecting iteration with orthogonalization
    if ( Dc is not None ):
      if ( Ds is not None ): Omega = Dc[:, U] - Psi_n @ ( Psi.T @ Dc[:, U] ) # orthogonalize Dc w.r.t Ds
      else: Omega = Dc[:, U] # If no imposed term start with unorthogonalized dictionary
      
      # ---------------------------------------------------- 2. First iteration treated separately since no orthogonalization -----------------------------------------------------------
      n_Omega = HF.Norm2( Omega ) # norm squared or fudge factor elementwise
      ell = tor.argmax( ( Omega.T @ y )**2 / tor.ravel( n_Omega ) ) # get highest QERR index

      if ( self.MorphDict is not None ): # Morphing necessary
        MOut = Morphing.Morpher( U, ell, Psi, Psi_n, y, A, W, L, Ds, Dc, self.MorphDict )
        
        if ( MOut is not None ): # Function has been morphed
          self.MorphDict["MorphData"].append( MOut[0] )
          self.MorphDict["DcNames"].append( MOut[2] ) # Parse Morphing Output  
          L.append( Dc.shape[1] ) # append newly added Regressor
          Dc = tor.column_stack( ( Dc, MOut[1] ) ) # append morphed term to dictionary ( unorthogonalized but centered )
          Reg = MOut[1] - Psi_n @ ( Psi.T @ MOut[1] ) # orthogonalize, since previous terms exist
          n_Reg = HF.Norm2( Reg )

      if ( ( self.MorphDict is None ) or ( MOut is None ) ): # no morphing, since either 1 ) deactivated or 2 ) not morphable    
        L.append( U[ell] ) # no morphing, so store the actual index
        Reg = Omega[:, ell] # selected regressor
        if ( len( U ) > 1 ): n_Reg = n_Omega[0, ell].item() # selected Regressor norm
        else: n_Reg = n_Omega # TODO check what's going on # must be an array not an int, holds for length zero and 1

      # Common to morphed and non-morphed
      if ( Ds is not None ): # if not 0th regression term
        A.append( Psi_n.T @ Dc[:, L[-1]] ) # Multiply all normed orthogonalized regressors with the currently chosen unorthogonalized regressor
        s +=1 # increment A-matrix column count

      Psi = tor.column_stack( ( Psi, Reg ) ) # normed matrix ( if Ds is None, Psi is empty and this is just a reshape )
      Psi_n = tor.column_stack( ( Psi_n, Reg / n_Reg ) ) # normed matrix ( if Ds is None, Psi_n is empty and this is just a reshape )
      W.append( (Psi_n[:,-1] @ y).item() ) # add orthogonal regression coefficient, extract value from arrray with
      ERR.append( W[-1]**2 * n_Reg / s2y ) # store this regressors ERR
      U.remove( U[ell] ) # update unused indices list
      if ( self.verbose ): ProgressCount.update() # Update counter

      # ------------------------------------------------------------------------ 3. Optimal regressor search loop ---------------------------------------------------------------------------
      while ( ( 1 - np.sum( ERR ) > tol ) and ( s < MatSize ) ): # while not enough variance explained ( empty lists sum to zero ) and still regressors available ( nS + nC )
        
        if ( ( self.Abort and ( s > MaxTerms ) )): # for leaf nodes only
          return ( tuple( [np.array( L, dtype = np.int64 ) ]) ) # R[1/4] this returns a valid L, which is however disqualified as solution candidate being already too long
        
        if ( LI is not None ):
          RegIdx = SameStart( np.concatenate( ( LI, L ) ), self.LookUpDict ) # check if regression is to be continued (doesn't contain Ds content, since no index)
          if ( RegIdx != [] ): # Match found, so exit
            self.AbortedRegs += 1 # OOIT - for statistics not needed for regression
            return ( np.setdiff1d( self.LG[RegIdx], LI, True ), RegIdx ) # R[2/4] eliminate LI from predicted L, since LI is concatenated by arbo
        
        # 1. Computations
        Omega = HF.DeleteColumn( Omega, ell ) - Psi_n[:,-1, None] @ ( Psi[:, -1, None].T @ Dc[:, U] ) # Single Gram Schmidt on unused regressors ( parenthesis avoid outerproduct broadcast )
        n_Omega = HF.Norm2( Omega ) # squared euclidean norm of Omega, Omega_n not stored to avoid recomputation since practically entire dictonary
        ell = tor.argmax( tor.ravel( ( Omega.T @ y )**2 / n_Omega ) ) # store index of highest QERR regressor

        if ( self.MorphDict is not None ): # Morphing needs to be done
          MOut = Morphing.Morpher( U, ell, Psi, Psi_n, y, A, W, L, Ds, Dc, self.MorphDict ) # s+1 since those entries are being written in and must thus be included
          
          if ( MOut is not None ): # Function has been morphed
            self.MorphDict["MorphData"].append( MOut[0] )
            self.MorphDict["DcNames"].append( MOut[2] ) # Parse Morphing Output
            L.append( Dc.shape[1] ) # append newly added Regressor
            Dc = tor.column_stack( ( Dc, MOut[1] ) ) # append morphed term to dictionary ( unorthogonalized )
            Reg = MOut[1] - Psi_n @ ( Psi.T @ MOut[1] ) # orthogonalize ( no reshape needed )
            n_Reg = HF.Norm2( Reg )  # squared euclidean norm of Reg or fudge factor

        if ( ( self.MorphDict is None ) or ( MOut is None ) ): # no morphing, since either 1 ) deactivated or 2 ) not morphable
          L.append( U[ell] ) # add the unmorphed = original term to the regression
          Reg = Omega[:, ell] # selected regressor
          if ( len( U ) > 1 ): n_Reg = n_Omega[0, ell].item() # selected Regressor norm
          else: n_Reg = n_Omega # TODO check what's going on # must be an array not an int, holds for length zero and 1
          # else: n_Reg = tor.tensor( [[n_Omega]] ) # must be an array not an int, holds for length zero and 1

        # 2. Data storage
        A.append( Psi_n.T @ Dc[:, L[-1]] ) # Multiply all normed orthogonalized regressors with the currently chosen unorthogonalized regressor
        Psi = tor.column_stack( ( Psi, Reg ) ) # unnormed matrix
        Psi_n = tor.column_stack( ( Psi_n, Reg / n_Reg ) ) # normed matrix
        W.append( ( Psi_n[:, -1] @ y ).item() ) # store orthogonal regression coefficient ( w )
        ERR.append( W[-1]**2 * n_Reg / s2y ) # store this regressors ERR, eliminating the y[k-i] terms is done in the return
        U.remove( U[ell] ) # update unused indices list
        if ( U == [] ): print( "\n\nThe entire dictionary has been used without reaching the desired tolerance!\n\n" )
        s += 1 # increment the A-matrix column count
        if ( self.verbose ): ProgressCount.update() # Update counter
    
    # ------------------------------------------------------------------------ 4. Output generation ---------------------------------------------------------------------------
    if ( self.verbose ): ProgressCount.close() # close tqdm counter

    if ( ( s > MaxTerms ) or ( OutputAll == False ) ):
      return ( tuple( [ np.array( L, dtype = self.INT_TYPE ) ]) ) # R[3/4] only triggered if MaxTerms overwritten, L → int, since if self.Dc=None then L=[] is assuemd float
    else:
      # L casted to int for the Dc=None case where L=[] is assuemd float. Return only Dc since MorphData and DcNames are overwritten internally
      return ( HF.SolveSystem( A, W ).cpu().numpy(), np.array( L, dtype = self.INT_TYPE ), np.array( ERR ) ) # R[4/4] return regression coeffs theta and used regressor names ( only selected from Dc )



  # *********************************************************************************************** Actual ADMOrLSR Algorithm **********************************************************************************
  def fit( self, FileName = None, SaveFrequency = 0 ):
    '''Breadth-first Search Arborescent rFOrLSR → returns theta, L and ERR in the order in which the regressors appear in the data matrix
    
    ### Inputs:
    - `FileName`: ( str ) FileName to the pickle file ( default: None )
    - `SaveFrequency`: ( int ) frequency in seconds of saving the Arborescence content ( default: 0 for no saving)
    
    ### Output:
    - `theta`: ( ( nr, )-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( ( nr, )-sized string nd-array ) containing the regressors indices
    - `ERR`: ( ( nr, )-sized float nd-array ) containing the regression's ERR
    '''

    if ( self.y is None ): raise AssertionError( "y is None meaning that an uninitialized Arborescence is being loaded" )
    if ( SaveFrequency != 0 ): self.SaveFrequency = SaveFrequency * 60 # overwrite if given
    if ( FileName is not None ): self.FileName = FileName # overwrite if given

    # --------------------------------------------------------------------------------- Traversal Init ----------------------------------------------------------------------------
    # Root Computation
    if ( self.Q.empty() and ( self.NotSkipped == 0 ) ): # only true if Arbo non-initialized (or if someone loads an ended arbo from a file, but what's the point?)
      print( "Performing root regression" )
      theta, L, ERR = self.rFOrLSR( self.y, self.Ds, self.Dc, None, self.tol1, OutputAll = True ) # create new temp data, use classe's Dc
      self.NotSkipped = 1 # Root is always computed
      if ( ( self.MaxDepth == 0 ) or ( self.Dc is None ) ):
        self.theta = theta; self.L = L; self.ERR = ERR # Write result into Arbo object
        return ( theta, L, ERR, self.MorphDict, self.Dc ) # Stop at level 0 which is the root, extract only element in the list

      self.LG.append( np.array( L, dtype = self.INT_TYPE ) ) # add Root with 1 index to lookup dictionary (:1 needed to keep it as array)
      for i in range( len( L ) ): # add all lenghts over the current level (the reg will never ask less since those terms are imposed)
        self.LookUpDict[ tuple( np.sort( L[:i+1] ) ) ] = 0 # append the sorted list to lookup dictionary with self.LG's last index
      
      self.MinLen = [ len( L ), len( L ) ] # list so that the MinLen per level is stored, twice since once for root then for first level
      self.MaxDepth = min( ( self.MaxDepth, self.MinLen[-1] ) ) # update to implement ADTT
      self.nTermsNext = 0 # number of terms in the next level, zero since unknown here
      print( "Shortest encountered sequence (root):", self.MinLen[-1], '\n' )

      for ImposedReg in L: self.Q.put( np.array( [ ImposedReg ], dtype = self.INT_TYPE ) ) # append all imposed regressors to the Queue as one element lists
      self.nTermsCurrent = len( L ) # update current level number of terms
    
    ProgressBar = tqdm.tqdm( total = self.nTermsCurrent, desc = f"Arborescence Level { len( self.Q.peek() ) }", unit = " rFOrLSR" )  # Initialise progrssbar with known total number of iterations of first level 
    ProgressBar.update( self.nComputed ) # insert the correct progress
    StartTime = timeit.default_timer() # start time counter for memory dumps


    # --------------------------------------------------------------------------------- Arborescence Traversal ----------------------------------------------------------------------------
    while ( not self.Q.empty() ):
      LI = self.Q.get() # retrieve the next FIFO element

      if ( ( self.SaveFrequency != 0 ) and ( timeit.default_timer() - StartTime > self.SaveFrequency ) ):
        self.MemoryDump( ProgressBar.n ) # Backup
        StartTime = timeit.default_timer() # reinitialize counter here to not count the backup time
      
      if ( len( LI ) > self.MinLen[-1] ): break # no self.nS is used since also ignored in self.MinLen
      self.TotalNodes += 1 # total number of traversed nodes

      ellG = SameStart( LI, self.LookUpDict ) # Is this regression already computed? Yes = return index, else [] if not

      if ( ellG == [] ): # This regression isn’t equal to a previous one, else pass and use returned index
        self.NotSkipped += 1 # just for statistics not needed for regression

        U_tmp = self.U.copy() # allow entire dictionary, copy to avoid overwrite and recreatino each time
        for li in LI: 
          if ( li < self.nC ): U_tmp.remove( li ) # remove imposed indices, only not morphing terms which are not present anyways ( they are however imposed, see above )

        Output = self.rFOrLSR( self.y, tor.column_stack( ( self.Ds, self.Dc[:, LI.astype( np.int64 )] ) ), self.Dc, U_tmp, self.tol2, MaxTerms = self.MinLen[-1] + self.nS, LI = LI ) # rFOrLSR with Li imposed
        # ellG doesn't contain LI if retrieved from an OOIT prediction, which is however not a problem


        ellG = np.concatenate( ( np.sort( LI ), Output[0] ) ) # sort the current level first terms (len( LI) )
        if ( len( Output ) == 1 ): # Regression was computed and not OOIT-predicted
          self.LG.append( np.array( ellG, dtype = self.INT_TYPE ) ) # unknown regression, so store it
          Value = len( self.LG ) - 1 # current regression index is the last LG entry ( -1 since zero-based )

        
        else: Value = Output[1] # regression was OOIT-predicted

        for i in range( len( LI ), len( ellG ) ): # add all lenghts over the current level (the reg will never ask less since those terms are imposed)
          key = tuple( np.sort( ellG[:i] ) ) # make the key permutaion invariant
          if ( ( key not in self.LookUpDict ) or ( len( self.LG[ self.LookUpDict[key] ] ) > len( ellG ) ) ): # if not an already known reg or if this one is shorter, overwrite
            self.LookUpDict[ key ] = Value # append the sorted list to lookup dictionary with self.LG's last index
        
        self.MinLen[-1] = min( ( len( ellG ), self.MinLen[-1] ) ) # update shortest known reg. sequence. Use only L since already concatenated to LI
        self.MaxDepth = min( ( self.MaxDepth, self.MinLen[-1] ) ) # update to implement ADTT
      
      else: ellG = self.LG[ ellG ] # SameStart returned an index, so a match was found: retrieve it

    
      if ( len( LI ) < self.MaxDepth ): # Iterate next level? (new regressions are only added by the previous level: if len(LI) == self.MaxDepth, AorLSr is in the last level → don't add terms for the next one)
        LC = AllCombinations( LI, ellG, self.INT_TYPE ) # compute all term combinations, pass INT_TYPE since not a class member function
        self.nTermsNext += LC.shape[0] # add set cardinality to nTerms of the next level
        for combinations in LC: self.Q.put( combinations ) # append all imposed regressors to the Queue
      
      ProgressBar.update() # increase count, Done here since almost all processing is done for this node, must be before the if, since this handles the level change
      
      if ( len( LI ) < len( self.Q.peek() ) ): # reached last node in level?
        ProgressBar.close() # close previous one to avoid glitches before creating new one below ( if not next level )
        print( "Shortest encountered sequence:", self.MinLen[-1], '\n' )

        # When new level is hit, the key of the just traversed length are no longer needed, since LI is longer and thus no requests of that length will ever be made again
        for reg in list( self.LookUpDict.keys() ): # Act on the keys, list since deleting while iterating is illegal
          if ( len( reg ) <= len( LI ) ): del self.LookUpDict[reg] # remove the entry
        
        if ( len( LI ) + 1 <= self.MaxDepth ): # Is next level to be iterated?
          ProgressBar = tqdm.tqdm( total = self.nTermsNext, desc = f"Arborescence Level {len( LI ) + 1}", unit = " rFOrLSR" )# Initialise progressbar with number of iterations
          self.nTermsCurrent = self.nTermsNext # number of node in this level (needed to load the progressbar from file)
          self.MinLen.append( self.MinLen[-1] ) # new entry for the new level

        self.Abort = len( LI ) + 1 == self.MaxDepth # early exit non-competitive regressions since this level contains the leaves ( only true at maxdepth )
        
        self.nTermsNext = 0 # reinitialize next level nodes counter ( must be done after the use for the progressbar )
    
    ProgressBar.close() # close previous one to avoid glitches before creating new one below ( if not next level )
    
    print( "Finished Arborescence traversal. Shortest encountered sequence:", self.MinLen[-1], '\n' )
    
    if ( self.SaveFrequency > 0 ): self.MemoryDump( ProgressBar.n ) 
    return ( self.validate() ) # call validation memeber function itself calling the user vaildation function

  # ********************************************************************************** Best Model selection / Validation **********************************************************************************
  def validate( self ):
    ''' "least terms = selected" heuristics from iFOrLSR paper wich adds a selection based on the minimal MAE.
    Note: This function can be used to get intermediate results during an arborescence traversal since the search data is not overwritten.

    ### Outputs:
    - `theta`: ( ( nr, )-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( ( nr, )-sized string nd-array ) containing the regressors indices
    - `ERR`: ( ( nr, )-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary
    
    '''
    print( "Starting Validation procedure." )
    MinError = tor.inf; Processed = [] # Variables storing the shortest sequence with Min Validation ’s position 

    for reg in tqdm.tqdm( self.LG, desc = "Validating", unit = " Regressions" ): # and over each regressor sequence
      if ( len( reg ) == self.MinLen[-1] ):
        reg = np.sort( reg ) # sort regressor sequence
        if ( tuple( reg ) not in Processed ): # compute error metric only if candidate to best regression.
          Processed.append( tuple( reg ) ) # add to processed list, cast to tuple since ndarrays can't use "in"
          theta, _, ERR, = self.rFOrLSR( self.y, tor.column_stack( ( self.Ds, self.Dc[:, reg.astype( np.int64 )] ) ), None, None, None, OutputAll = True )
          
          if ( self.V is None ): Error = 1 - np.sum( ERR ) # take ERR if no validation dictionary is passed
          else: Error = self.F( theta, reg, ERR, self.V, self.MorphDict, self.DcFilterIdx ) # compute the passed custom error metric

          if ( Error < MinError ): MinError = Error; self.theta = theta; self.L = reg.astype( np.int64 ); self.ERR = ERR # update best model

    print( f"\nValidation done on { len( Processed ) } different Regressions. Best validation error: { float( MinError ) }\nAOrLSR sucessfully terminated\n" )
    
    print( "\nOut of", self.TotalNodes,"only", self.NotSkipped, "regressions were computed, of which", self.AbortedRegs, "were OOIT - aborted.\n" )

    return ( self.theta, self.L, self.ERR, self.MorphDict, self.Dc )


  # *************************************************************************** Helper function displaying the regression results ***********************************************************************
  def get_Results( self ):
    if ( self.theta is None ): raise AssertionError( "No regression results available, thus the fitting hasn't been finished. To abort the arbo traversal, use Arbo.validate()" )
    return ( self.theta, self.L, self.ERR, self.MorphDict, self.Dc ) # same as the fit() function
  

  # *************************************************************************** Helper function changing the Arbo depth if possible ***********************************************************************
  def set_ArboDepth( self, Depth ):
    '''Performs the necessary verifications before changing the Arbo depth'''
    # The order of the following checks is important to output the real error, as conditions overlap.
    if ( Depth < 0 ):                    raise AssertionError( "Depth can't be negative" )
    if ( Depth < len( self.Q.peek() ) ): raise AssertionError( f"The Arbo has already passed that depth. The current depth is { len( self.Q.peek() ) }" )
    if ( self.Abort ):                   raise AssertionError( "The Arbo is already in its last level. It can't be made deeper since leaf nodes don't produce children" )
    
    if ( Depth > self.MinLen[-1] ):
      print (f"\n\nWarning: Desired depth is greater than the shortest known sequence ({ self.MinLen[-1] }). Thus no update was performed.\n\n" )
      return # prevents overwriting by early exiting the function.
    
    self.MaxDepth = Depth # nothing talks against it, if we arrived until here :D


  # *************************************************************************** Helper function displaying the regression results ***********************************************************************
  def PlotAndPrint( self, PrintRegressor = True ):
    '''RegNames must contain all regressor names ( Ds and Dc and morphing ) in the same order as theta'''
    if ( self.L is None ): raise AssertionError( "The fitting hasn't been finished, thus no regression results are available. To get intermediate results, trigger the validation procedure" )
    
    if ( self.Dc is not None ): # results in the problem of self.L.shape == (0,)
      RegNames = np.concatenate( (self.DsNames, np.ravel( self.DcNames[self.L] ) ) )
      yHat = tor.column_stack( ( self.Ds, self.Dc[:, self.L] ) ) @ tor.Tensor( self.theta ) # Model prediction on the training data
    else: # only a Ds exists
      RegNames = self.DsNames
      yHat = self.Ds @ tor.Tensor( self.theta ) # Model prediction on the training data
    
    yNorm = tor.max( tor.abs( self.y ) ) # Compute Model, normign factor to keep the display in % the the error
    Error = ( self.y - yHat ) / yNorm # divide by max abs to norm with the max amplitude
    
    # Metrics
    MedianAbsDeviation = tor.median( tor.abs( Error - tor.median( Error ) ) ).item() # Outlier stable ( median + abs instead of mean + ()² ) variance-type of metric
    AbsError = tor.abs( Error )
    MeanAbsErrorPercent = 100 * tor.mean( AbsError ).item()
    
    # String formatting
    MeanAbsErrorStr = '{:.3e}'.format( MeanAbsErrorPercent ) if ( MeanAbsErrorPercent < 0.001 ) else str( round( MeanAbsErrorPercent, 3 ) )
    MaxDeviationStr = '{:.3e}'.format( 100 * tor.max( AbsError ) ) if ( 100 * tor.max( AbsError ) < 0.001 ) else str( tor.round( 100 * tor.max( AbsError ), decimals = 3 ).item() ) # max is stored in array so cheap
    MedianAbsDerivationStr = '{:.4e}'.format( 100 * MedianAbsDeviation ) if ( 100 * MedianAbsDeviation < 0.001 ) else str( round( 100 * MedianAbsDeviation, 3 ) )
    

    # A) Ground thruth and fitting error plot
    Fig, Ax = plt.subplots( 2, sharex = True )
    Ax[0].plot( ( self.y / yNorm ).cpu(), "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax[0].plot( ( yHat / yNorm ).cpu(), "tab:orange", marker = '.', markersize = 5 ) # force default orange
    Ax[0].legend( ["System Output y[k]", "Estilmation $\hat{y}$[k]"] )
    Ax[0].grid( which = 'both', alpha = 0.5 )

    Ax[1].plot( Error.cpu(), "#00aaffff", marker = '.', markersize = 5 )
    Ax[1].set_xlim( [0, len( self.y )] )
    Ax[1].set_title( str( len( self.L ) ) + " Terms yielding MAE: " + MeanAbsErrorStr + "%. Max dev.: " + MaxDeviationStr + "%. MAD: " + MedianAbsDerivationStr + "%" )
    Ax[1].legend( ["$y[k]-\hat{y}[k]$"] )
    Ax[1].grid( which = 'both', alpha = 0.5 )
    Fig.tight_layout() # prevents the plot from clipping ticks

    
    # B) ERR stem plots with MAE reduction: this plot is slightly meaning-less since the ERR and the MAE reduction of each term depends on their position in the BVS
    # Also their order in the BVS is arbitrary since imposeed by the AOrLSR but whatever
    
    # Arg Sort ERR and impose same order on L
    MAE = [] # list containing the MAE values from the models progressively build up
    Order = np.flip( np.argsort( self.ERR ) )
    ERRtemp = self.ERR[Order]; Sortedtheta = self.theta[Order]; RegNames = RegNames[Order] # impose same order on all datastructures
    if ( self.Dc is not None ):
      Ltemp = self.L[Order]; # L is an empty array is no Dc is passed
      Imposed = tor.column_stack( ( self.Ds, self.Dc[:, Ltemp] ) )#  invalid indexation is Dc is empty or None
    else: Imposed = self.Ds

    for i in range( 1, Imposed.shape[1] + 1 ): # iterate over the index list of all selected terms. +1 since the indexing excludes the last term
      theta_TMP = self.rFOrLSR( self.y, Ds = Imposed[:, :i], OutputAll = True )[0] # only impose the terms to get the new theta
      MAE.append( ( tor.mean( tor.abs( self.y - Imposed[:, :i] @ tor.Tensor( theta_TMP ) ) ) / yNorm ).item() )
    
    Fig2, Ax2 = plt.subplots( 2 ) # 2 because the first Ax object is outputted by the function
    Ax2[0].set_title( "Top: All " + str( len( ERRtemp ) ) + " Regressors' ERR in descending oder. Bottom: Model MAE evolution" )
    Ax2[0].stem( ERRtemp, linefmt = "#00aaffff" )
    plt.setp( Ax2[0].get_xticklabels(), visible = False ) # desactivate the ticks
    Ax2[0].set_ylim( [0, 1.05 * max( ERRtemp )] )
    Ax2[0].grid( axis = 'y', alpha = 0.5 )
    
    Ax2[1].plot( MAE, "#00aaffff", marker = '.', markersize = 5 )
    Ax2[1].set_xticks( np.arange( len( ERRtemp ) ), RegNames, rotation = 45, ha = 'right' ) # setting ticks manually is more flexible as it allows rotation
    Ax2[1].grid( axis = 'x', alpha = 0.5 )
    for i, v in enumerate( MAE ): Ax2[1].text( i, v + 0.1 * max( MAE ), '{:.3e}'.format( v ), ha = "center" ) # print exact values
    Ax2[1].set_ylim( [0, max( MAE ) * 1.3] )
    
    Fig2.tight_layout() # prevents the plot from clipping ticks
    plt.subplots_adjust( hspace = 0.001 )
    
    # print Summary to console
    print( Imposed.shape[1], "Terms yielding an Mean absolute Error (MAE) of", MeanAbsErrorStr,"% and a maximal deviation of", MaxDeviationStr +
          "% and a Median Absolute Deviation (MAD) of", MedianAbsDerivationStr )
    
    if ( PrintRegressor ): # print the regressors in a readable manner
      print( "\nRecognized terms:" )
      for i in range( len( Sortedtheta ) ): print( Sortedtheta[i], RegNames[i] )
      print( "\n" )

    return ( ( Fig, Fig2 ), ( Ax, Ax2 ) ) # for further operations