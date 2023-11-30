# ######################################################################################## Imports ###################################################################################
# Math
import numpy as np
import torch as tor
import matplotlib.pyplot as plt

# Rest of the Lib: "from ."" is same Folder and ".Name" is subfolder import
from .HelperClasses import Queue, MultiKeyHashTable # for the BFS Arborescent Regression
from .HelperClasses.NonLinearity import NonLinearity # import here to give acces to the user (add to namespaec rFOrLSR)
from .CTors import __init__ # import all constuctors and helper functions
from . import Morphing
from . import HelperFuncs as HF

import dill # memory dumps
import tqdm # progress bars
import timeit # time measurements for Back-ups

device = HF.Set_Tensortype_And_Device() # force 64 bits, on GPU if available
Identity = NonLinearity( "id", lambda x: x ) # pre-define object for user


# ############################################################################ Variable Selection Procedure #############################################################################
def MaxLagPlotter( x, y, MaxLags = ( 15, 15 ), MaxOrder = 5, VarianceAcceptThreshold = 0.98, Plot = True ):
  '''Variable selection function determining the maximum lags for y and x for rFOrLSR dictionary sparcification.
  This function is NARMAX specific since lagged variables are checked using arbitrary-order polynomial NARX models rather than Taylor expansions.
  Everything in purple on the plot is below the VarianceAcceptThreshold.
  
  Note: For model order > 2, this function might be a lot slower than an Arbo with a large dictionary, so use only for analysis or for Dcs not fitting in memory.
  
  ### Inputs:
  -`x`: (1D torch.Tensor) containing the system input
  -`y`: (1D torch.Tensor) containing the system output signal
  -`MaxLags`: (2D int Tuple) containing the maximum lags for n_b and n_a, defaults to 30 for both ( n_b, n_a )
  -`MaxOrder`: (int > 0) Maximum approximation order used for the estimation
  -`VarianceAcceptThreshold`: ( float ) the minimum explained variance of the NARMAX expansion to estimate the needed delays
  -`Plot`: (bool) if True, the plot will be generated
  
  ### Output:
  - `Modelorder`: (int) The chosen expansion's order, to be passed as (monomial) ExpansionOrder parameter to RegressionMatrix ( Dictionary CTor )
  - `Grid`: ( MaxLags[0], MaxLags[1] )-shaped np.array containing the ERR values displayed by the plot
  - `Recommendations`: (Dict) containing the optimal lags with the system with the minimal x & y, x, y lags.
  '''
  
  # Bullshit prevention:
  if ( x.shape != y.shape ):                          raise AssertionError( "x and y must have the shape shape.Note that both are flattened for prcessing" )
  if ( ( x.ndim != 1 ) or ( y.ndim != 1 ) ):          raise AssertionError( "x or y is not a (p,)-shaped Tensor" )
  if ( ( MaxOrder < 1 ) or ( not isinstance( MaxOrder, int ) ) ): raise AssertionError( "MaxOrder must be an int >= 1" )
  
  # --------------------------------------------------------------------------------- Ds only FOrLSR ------------------------------------------------------------------------------
  y = tor.ravel( y )
  x = tor.ravel( x )

  def MiniFOrLSR ( y, Ds ):
    ''' Very minimal Version of FOrLSR, as only the ERR of imposed terms is necessary '''
    s2y = ( y @ y ).item() # mean free observation empiric variance
    ERR = [] # list of Error reduction ratios of all selected regressors ( Dc and Ds )
    Psi = tor.empty( ( len( y ), 0 ) ); Psi_n = tor.empty( ( len( y ), 0 ) ) # Create empty ( p, 0 )-sized matrices to simplify the code below

    # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
    Psi = Ds[:, 0, None] # unnormed orthogonal regressor matrix ( already centered ) reshaped as column
    n_Omega = HF.Norm2( Psi ) # squared euclidean norm of Omega or fudge factor
    Psi_n = Psi / n_Omega # normed orthogonal regressor matrix
    ERR.append( ( ( Psi_n.T @ y ).item() )**2 * n_Omega / s2y ) # W[-1]^2 * n_Omega/s2y ) as usual but without storing W
    
    for col in range( 1, Ds.shape[1] ): # iterate over columns, start after position 1
      if ( np.sum( ERR ) >= 1 ): return ( 1.0 ) # R[1/2] early exit if max ERR reached
      # Computations
      Omega = Ds[:, col] - Psi_n @ ( Psi.T @ Ds[:, col] ) # orthogonalize only the current column ( no reshape needed )
      n_Omega = HF.Norm2( Omega )  # squared euclidean norm of Omega or fudge factor
      ERR.append( ( ( Omega @ y ).item() / n_Omega )**2 * n_Omega / s2y ) # W[-1]^2 * n_Omega/s2y ) as usual but without storing W
      
      # Data storage, add current regressor
      Psi = tor.column_stack( ( Psi, Omega ) ) # unnormed matrix
      Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) ) # normed matrix
    
    return ( np.sum( ERR ) ) # R[2/2]

  # --------------------------------------------------------------------------------- A) Model order evaluation ------------------------------------------------------------------------------
  ModelOrder = 0 # try linear model first, incremented in the while to 1
  ModelExplainedVariance = 0 # Init to zero since not computed

  print( "Estimating model order." )
  while ( 5 ): # if less than Minvariance variance is explained, redo the analysis with a higher order model, since maxlag= max variance
    ModelOrder += 1 # increase order
    y_cut, RegMat, RegNames = CTors.Lagger( ( x, y ), MaxLags ) # construct linear regressor matrix, ignore RegNames return
    RegMat, RegNames = CTors.Expander( RegMat, RegNames, ExpansionOrder = ModelOrder )

    ModelExplainedVariance = MiniFOrLSR( y_cut - y_cut.mean(), RegMat - RegMat.mean( axis = 0, keepdims = True ) ) # pass y separately and exclude it in Ds, then sum of all ERRs
    
    if ( ( ModelOrder > MaxOrder + 1 ) or ( ModelExplainedVariance > VarianceAcceptThreshold ) ): break # do while condition, +1 since next iteration will exceed limit
    
  print( "\nA order", ModelOrder, "model explaining", 100 * ModelExplainedVariance, "% of the variance was selected. Computing the plot:" )
  
  # --------------------------------------------------------------------------------- B) Model computation ------------------------------------------------------------------------------
  Grid = np.full( ( MaxLags[1] + 1, MaxLags[0] + 1 ), np.nan ) # y's are rows and the x's columns to have a correct graph orientation, +1 due to x[k], y[k]
  ProgressBar = tqdm.tqdm( total = Grid.size ) # Initialise progressbar while declaring total number of iterations

  # Unintuitively, it's the MiniFOrLSR which accounts for 99.99% of the time, not RegressorMatrix, so optimizing that out isn't of interest.
  # Also y_cut has a different length at each iteration so it can't be stored
  for na in range( MaxLags[1] + 1 ): # iterate over y values, +1 to contain the end-of-range value
    for nb in range( MaxLags[0] + 1 ): # iterate over x values, +1 to contain the end-of-range value
      
      if ( ( Grid[ max( na-1, 0 ), nb] == 1.0 ) and ( Grid[ na, max( nb-1, 0 )] == 1.0 ) ): 
        Grid[ na, nb ] = 1.0 # if both previous regressor lists are already sufficient to achive full precision, don't recompute unnecessarily everything
      
      else:
        y_cut, RegMat, RegNames = CTors.Lagger( Data = ( x, y ), Lags = ( nb, na ) ) # construct linear regressor matrix, ignore RegNames return
        RegMat, RegNames = CTors.Expander( RegMat, RegNames, ExpansionOrder = ModelOrder )
        
        Grid[na, nb] = MiniFOrLSR( y_cut - y_cut.mean(), RegMat - RegMat.mean( axis = 0, keepdims = True ) ) # store the ERR only

      ProgressBar.update() # increase count
  ProgressBar.close() # Necessary
  
  # --------------------------------------------------------------------------------- C) Lags recommendation -----------------------------------------------------------------------------------
  Recommendations = { "Min_XY": (MaxLags[1], MaxLags[0]), # Position with the smallest x and y lags
                      "Min_X":  (MaxLags[1], MaxLags[0]), # Position with the smallest x lag
                      "Min_Y":  (MaxLags[1], MaxLags[0]), # Position with the smallest y lag
                    }
  
  BestIdx = np.inf

  for na in range( MaxLags[1] + 1 ):
    for nb in range( MaxLags[0] + 1 ):
      
      if ( Grid[ na, nb ] == 1.0 ): # valid solution
        if ( na + nb < BestIdx ): # Smallest a+b position. Taking <= would allow more y terms in, yielding less numereically stable solution systems
          BestIdx = na + nb
          Recommendations["Min_XY"] = (na, nb)

  for na in range( MaxLags[1] + 1 ): # iterate over y values first, since we're looking for the smallest nb with the smallest na
    BestIdx = np.argmax( Grid[ na, : ] == 1.0 ) # find the first nb == 1. Guaranteed to be max since results are clipped inside MiniFOrLSR
    if ( BestIdx != 0 ): Recommendations["Min_Y"] = (na, BestIdx); break

  for nb in range( MaxLags[0] + 1 ): # iterate over x values first, since we're looking for the smallest na with the smallest nb
    BestIdx = np.argmax( Grid[ : , nb ] == 1.0 ) # find the first na == 1. Guaranteed to be max since results are clipped inside MiniFOrLSR
    if ( BestIdx != 0 ): Recommendations["Min_X"] = (BestIdx, nb); break
        

  # ---------------------------------------------------------------------------------- D) Plot -------------------------------------------------------------------------------------------------
  if ( Plot ):
    plt.pcolormesh( Grid, cmap = 'viridis', edgecolors = 'k', linewidth = 2, vmin = VarianceAcceptThreshold, vmax = 1 ) # Everything below the VarianceAcceptThreshold is not of interest
    Ax = plt.gca()

    DotSize = 50
    Ax.scatter( Recommendations["Min_Y"][1],  Recommendations["Min_Y"][0],  color = 'r', s = DotSize ) # red
    Ax.scatter( Recommendations["Min_X"][1],  Recommendations["Min_X"][0],  color = 'b', s = DotSize ) # blue
    Ax.scatter( Recommendations["Min_XY"][1], Recommendations["Min_XY"][0], color = 'k', s = DotSize ) # black, last to be on top if multiple at same spot

    Ax.set_ylabel( "y[k-i] terms" ); Ax.set_xlabel( "x[k-i] terms" )
    Ax.set_aspect( 'equal' )
    plt.colorbar() # defaults to curernt Figure
    Ax.set_ylim( 0, MaxLags[1] + 1 ); Ax.set_xlim( 0, MaxLags[0] + 1 ) # Have both axis start at 0 in the bottom left corner (flips y-axis)
  
  return ( ModelOrder, Recommendations, Grid )
  

  
# ################################################################################### Default Validation procedure ####################################################################################
def DefaultValidation( theta, L, ERR, ValData, MorphDict, DcFilterIdx = None ):
  '''
  Default Validation function based on time domain MAE.
  
  ### Input
  - `theta`: ( (nr,)-shaped float nd-array ) containing the estimated regression coefficients
  - `L`: ( (nr,)-shaped int nd-array ) containing the regressors indices
  - `ERR`: ( (nr,)-shaped float nd-array ) containing the regression's ERR ( not used here but placeholder to adhere to AOrLSR standard )
  
  - `ValData`: ( dict ) containing the validation data to be passed to Dc's Ctors:
    → `ValData["y"]`: ( list of float torch.Tensors ) containing the system responses
    → `ValData["Data"]`: ( list of iterables of float torch.Tensors ) containing the data to be passed to the standard CTors ( Lagger, Expander, NonLinearizer )
    → `ValData["DsData"]`: ( list of float Torch.Tensors or None ) containing all imposed regressors column wise
    → `ValData["Lags"]`: ( 2D tuple of float ) containing the respective maximum delays for the passed Data
    → `ValData["ExpansionOrder"]`: ( int ) containing the monomial expansion's maximal summed power
    → `ValData["NonLinearities"]`: ( list of pointer to functions ) containing the regression's transformations to be passed to RegressorTransform
    → `ValData["MakeRational"]`: ( list of bool ) containing information about which functions are to be made rational
  
  - `MorphDict`: # ( dict ) # TODO (same as in __init__)

  - `MorphData`: ( list ) containing the requires information to recreate the morphed regressors
    → `index`: ( int ) containing the column number in Dc of the current regressor
    → `fs`: ( int ) containing the non-linearity's number
    → `LM`: ( list of int ) containing the indexes constituting the non-linearity's arguments
    → `ksi`: ( ( r, )-shaped Tensor ) containing the arguments linear combination's coefficients
  
  ### Output
  -`Error`: ( float ) containing the passed model's validation error on the validation set
  '''
  
  # --------------------------------------------------------------------  Bullshit prevention -----------------------------------------------------------------------
  if ( not isinstance( ValData, dict ) ): raise AssertionError( "The passed ValData datastructure is not a dictionary as expected" )
  
  for var in ["y", "Data", "DsData", "Lags", "ExpansionOrder", "NonLinearities", "MakeRational"]:
    if ( var not in ValData.keys() ): raise AssertionError( f"The validation datastructure contains no '{ var }' entry" )
  
  # Dirty conversion but simplifies the code a lot
  if ( ValData["DsData"] == [] ): ValData["DsData"] = None
  if ( ValData["Data"] == [] ):   ValData["Data"]   = None


  if ( not isinstance( ValData["y"], list ) ): raise AssertionError( "ValData's 'y' entry is expected to be a list of float torch.Tensors" )

  if ( ValData["Data"] is not None ):
    if ( not isinstance( ValData["Data"], list ) ):         raise AssertionError( "ValData's 'Data' entry is expected to be a list" )
    if ( len ( ValData["Data"] ) != len ( ValData["y"] ) ): raise AssertionError( "ValData's 'Data' and 'y' entries should have the same length" )

    for val in ValData["Data"]: # iterate over all passed Data tuples
      for reg in val: # Data entries contain unknown number of regressors in the MISO
        if ( not isinstance( reg, tor.Tensor ) ): raise AssertionError( "ValData's 'Data' entry is expected to be a list of 2D-tuples of float torch.Tensors" )

  if ( not isinstance( ValData["NonLinearities"], list ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )
  for func in ValData["NonLinearities"]: # iterate over all passed NonLinearities
    if ( not isinstance( func, NonLinearity ) ): raise AssertionError( "ValData's 'NonLinearities' entry is expected to be a list of NonLinearity objects" )
  
  if ( ValData["DsData"] is not None):
    if ( len( ValData["DsData"] ) != len( ValData["y"] ) ): raise AssertionError( "ValData's 'Data' and 'y' lists should have the same length" )

    for data in ValData["DsData"]: # iterate over all passed Data tuples
      if ( not isinstance( data[0], tor.Tensor ) ): raise AssertionError( "ValData's 'DsData' entry is expected to be a list of (torch.Tensor (for y), torch.Tensor (for Ds)) or None" )
      if ( not isinstance( data[1], tor.Tensor ) ): raise AssertionError( "ValData's 'DsData' entry is expected to be a list of (torch.Tensor (for y), torch.Tensor (for Ds)) or None" )

    for val in range ( len( ValData["DsData"] ) ):  # iterate over all passed Data tuples
      if ( ValData["y"][val].shape[0] != ValData["DsData"][val].shape[0] ): raise AssertionError( f"ValData's 'DsData' { val }-th entry has not the same length as teh { val }-th 'y' entry" )
          
  # -------------------------------------------------------------------------------  Validation computation loop -----------------------------------------------------------------------
  Error = 0 # total relative error

  # Estuimate the number of validations
  nValidations = 0
  if ( ValData["Data"] is not None ): nValidations = len( ValData["Data"] )
  if ( ValData["DsData"] is not None ):
    if ( ( nValidations != 0 ) and ( len( ValData["DsData"] ) != nValidations ) ): raise AssertionError( "ValData's 'DsData' entry is not the same length as 'Data'" )
    nValidations = len( ValData["DsData"] )
  if ( nValidations == 0 ): raise AssertionError( "No validation data was passed, as Data and DsData are both None or empty lists" )


  for val in range( nValidations ): # iterate over all passed Data tuples    
    # ----------------------------------------------------------------------------------- Handle Ds -----------------------------------------------------------------------------------
    if ( ValData["DsData"] is not None ):
      nS = ValData["DsData"][val].shape[1] # number of cols in Validation Ds
      yHat = ValData["DsData"][val] @ theta[ : nS ] # becomes a ( p, 1 )-shaped Tensor
    else: nS = 0; yHat = None # no Ds

    # ----------------------------------------------------------------------------------- Handle Dc -----------------------------------------------------------------------------------
    if ( ValData["Data"] is not None ):

      y, RegMat, RegNames = CTors.Lagger( Data = ValData["Data"][val], Lags = ValData["Lags"] ) # Create the delayed signal regressors

      if ( not tor.allclose( y, ValData["y"][val] ) ): raise AssertionError( "ValData['y'] and the y resulting from CTors.Lagger are not the same" )

      RegMat, RegNames = CTors.Expander( RegMat, RegNames, ValData["ExpansionOrder"] ) # Monomial expand the regressors
      RegMat, _, _ = CTors.NonLinearizer( y, RegMat, RegNames, ValData["NonLinearities"], ValData["MakeRational"] ) # add the listed regressors to the Regression matrix

      if ( DcFilterIdx is not None ): RegMat = RegMat[:, DcFilterIdx] # Filter out same regressors as for the regression

      if ( yHat is None ): yHat = tor.zeros( RegMat.shape[0] ) # no DsData passed
      elif ( ValData["DsData"][val].shape[0] != yHat.shape[0] ): raise AssertionError( "ValData['DsData'] and the resulting Dc are not of the same length" )
      
      # Centering
      Means = tor.mean( RegMat, axis = 0, keepdims = True ) # Store the means ( needed by the Morpher )
      RegMat -= Means # Center the regressors by subtracting their respecitve ( = columnwise ) means

      nC = RegMat.shape[1] # No added regressors to dict. nC needed to detect if morphed term or not

      for reg in range( len( L ) ): # iterate over all L and check if morphed
        if ( L[reg] >= nC ): raise AssertionError( "Morphing currently not supported" ) # morphed reg since index higher (or == since 0 based) than original Dc size
          
          # TODO adapt to Pytorch and to new morphing API, use ValData["y"][val] and center if y needed!
          # nMorphed = L[reg] - nC # number of the morphed term in the chronological list of creation
          # LM = MorphDict["MorphData"][nMorphed][2] # alias for clarity
          # f = MorphDict["fPtrs"][ MorphDict["MorphData"][nMorphed][1] ] # alias for clarity
          # Xl = RegMat[:, LM] + Means[:, LM]
          # fT = f( Xl @ MorphDict["MorphData"][nMorphed][-1] ); fT -= tor.mean( fT ) # MorphData[reg][-1]
          # yHat += theta[ nS + reg ] * fT

        else: yHat += theta[ nS + reg ] * RegMat[ :, L[reg] ] # normal, non-morphed regressor

    # handle y elegantly
    if ( ValData["y"][val].shape[0] != yHat.shape[0] ):
      if ( not tor.allclose( y, ValData["y"][val][ : -yHat.shape[0] ] ) ): # cut the front part of y which corresponds to the maximum delay
        raise AssertionError( f"ValData['y'][{ val }] was cut to fit the resulting estimate h_Hat's length, however the values differ. Please check that y variable" )
      y = ValData["y"][val][ : -yHat.shape[0] ]; y -= tor.mean( y )
    
    else: y = ValData["y"][val] - tor.mean( ValData["y"][val] ) # User passed the correct size

    Error += tor.mean( tor.abs( y - yHat ) / tor.mean( tor.abs( y ) ) ) # relative MAE
    
  return ( Error / nValidations ) # norm by the number of validation ( not necessary for AOrLSR but printed for the user )


# ####################################################################################################### AOrLSR Class ############################################################################################
class Arborescence:

  # ******************************************************************************************************** Init *************************************************************************************************
  def __init__( self, y = None, Ds = None, DsNames = None, Dc = None, DcNames = None, # Fitting Data
               tolRoot = 0.001, tolRest = 0.001, MaxDepth = 5, # Arbo size influencers: rho 1 & 2
               ValFunc = None, ValData = None, # Validation function and Data
               Verbose = False, # rFOrLSR progess feedback
               MorphDict = None, U = None,# Morphing related stuff
               FileName = None, SaveFrequency = 0 # Arbo Backup Parameters
              ):
    '''
    Class CTor. Can be called without any arguments if the object is supposed to be filled by the load() function.

    ### Input:
    → All arguments are optional, thus allowed to be None. If no argument is passed, an empty arbo is created and expected to be filled by Arbo.load().
    - `y`: ( None or (p,)-torch.Tensor ) System response
    - `Ds`: (None or (p, nS )-torch.Tensor ) Selected regressor matrix stacking (pre-) selected regressors column-wise
    - `DsNames`: (None or (nS,)-np.array of strings) Names of user pre-selected regressors
    - `Dc`: (None or (p, nC)-torch.Tensor ) Candidate regressor matrix stacking candidate regressors column-wise
    - `DcNames`: (None or (nC,)-np.array of strings) Names of candidate regressor matrices (only used by PlotAndPrint function)
    - `tolRoot`: (None or float) containing the maximum summed ERR threshold used for the root regression
    - `tolRest`: (None or float) containing the maximum summed ERR threshold used for all non-root nodes
    - `MaxDepth`: (None or int >= 0) containing the maximum number of levels to be iterated
    - `ValFunc`: (None or pointer to function) linking to the custom validation scalar function designed to work with the dict ValData
    - `ValData`: (None or dict) containing whatever data ValFunc needs to output the validation metric float
    - `Verbose`: (None or bool) containing whether to print the current state of the rFOrLSR (only meaningful for regressions with many terms)
    - `MorphDict`: (None or dict) containing the morphing parameters (see below for full expected content)
    - `U`: (None or list of int >=0) containing the indices of all Dc columns allowed as candidate regressors
    - `FileName`: (None or str) containing the path and file name to save the backups into
    - `SaveFrequency`: (None or float/int) containing the frequency in minutes at which to save the backups (10 = every 10 minutes)

    ### MorphDict content:
    - `NonLinMap`: (list of int) containing the non-linearity indices of all Dc columns allowed to be morphed (0 for no-morphing)
    ### TODO
    '''

    # ------------------------------------------------------------------------------------------ Type checks ---------------------------------------------------------------------------------

    if ( ( y is not None ) and ( not isinstance( y, tor.Tensor ) ) ):
      raise ValueError( "y must be None or a torch.Tensor" )

    if ( ( Ds is not None ) and ( not isinstance( Ds, tor.Tensor ) ) ):
      raise ValueError( "Ds must be None or a torch.Tensor" )

    if ( ( DsNames is not None ) and ( not isinstance( DsNames, np.ndarray ) or not all( isinstance( name, str ) for name in DsNames ) ) ):
      raise ValueError( "DsNames must be None or an np.array of strings" )

    if ( ( Dc is not None ) and ( not isinstance( Dc, tor.Tensor ) ) ):
      raise ValueError( "Dc must be None or a torch.Tensor" )

    if ( ( DcNames is not None ) and ( not isinstance( DcNames, np.ndarray ) or not all( isinstance( name, str ) for name in DcNames ) ) ):
      raise ValueError( "DcNames must be None or an np.array of strings" )

    if ( not isinstance( tolRoot, float ) ):
      raise ValueError( "tolRoot must be a float" )

    if ( not isinstance( tolRest, float ) ):
      raise ValueError( "tolRest must be a float" )

    if ( ( MaxDepth is not None ) and ( not isinstance( MaxDepth, int ) ) ):
      raise ValueError( "MaxDepth must be None or an int >= 0" )

    if ( ( ValFunc is not None ) and ( not callable( ValFunc ) ) ):
      raise ValueError( "ValFunc must be None or a callable function" )

    if ( ( ValData is not None ) and ( not isinstance( ValData, dict ) ) ):
      raise ValueError( "ValData must be None or a dict" )

    if ( ( Verbose is not None ) and ( not isinstance( Verbose, bool ) ) ):
      raise ValueError( "verbose must be None or a bool" )

    if ( ( MorphDict is not None ) and ( not isinstance( MorphDict, dict ) ) ):
      raise ValueError( "MorphDict must be None or a dict" )

    if ( ( U is not None ) and ( not isinstance( U, list ) or not all( isinstance( item, int ) for item in U ) ) ):
      raise ValueError( "U must be None or a list of integers >=0" )

    if ( ( FileName is not None ) and ( not isinstance( FileName, str ) ) ):
      raise ValueError( "FileName must be None or a str" )

    if ( ( SaveFrequency is not None ) and ( not isinstance( SaveFrequency, float ) ) and ( not isinstance( SaveFrequency, int ) ) ):
      raise ValueError( "SaveFrequency must be None a float or an int" )


    # ------------------------------------------------------------------------------------------ Store Passed Arguments ---------------------------------------------------------------------------------
    # Copy everything into the class

    if ( y is not None ): # Only when initializing empty arbo when loading from a file
      self.y = tor.ravel( y - tor.mean( y ) ) # flatski to (p,) vector for rFOrLSR
      if ( tor.isnan( tor.sum( self.y ) ) ): raise AssertionError( "Your system output y somehow yields NaNs, Bruh. We don't like that here" )
    else: self.y = None

    self.Ds = Ds
    self.Dc = Dc # columnwise centering and duplicate removal performed below
    self.DcMeans = None # Overwritten below, when Dc's non-redundant size (nC) is known after filtering duplicate regressors
    self.DcNames = DcNames
    self.DcFilterIdx = None # indexset of all non-duplicate regressors in Dc, computed below
    self.DsNames = DsNames
    self.tolRoot = tolRoot
    self.tolRest = tolRest
    self.MaxDepth = MaxDepth
    self.ValFunc = ValFunc
    self.ValData = ValData
    self.Verbose = Verbose
    self.MorphDict = MorphDict
    self.FileName = FileName
    self.SaveFrequency = SaveFrequency * 60 # transform from seconds into minutes
    self.INT_TYPE = np.int64 # default, if no DC present to overwrite it (thus not used since it only affects the arbo search)

    # ------------------------------------------------------------------------------------------ Argument Processing ---------------------------------------------------------------------------------

    # ~~~~~~~~~~~~ DC Stuff
    if ( self.DcNames is not None ): # needs to be checked first since used in Dc processing -if below
      if ( len( self.DcNames ) != self.Dc.shape[1] ): raise TypeError( "DcNames must be None or a np.array of the same length as Dc" )

    if ( self.Dc is not None ):
      if ( tor.isnan( tor.sum( self.Dc ) ) ): raise AssertionError( "Regressor Matrix Dc contains NaNs, Bruh. We don't like that here" )

      self.Dc, self.DcNames, self.DcFilterIdx = HF.RemoveDuplicates( self.Dc, self.DcNames )
      self.DcMeans = tor.mean( self.Dc, axis = 0, keepdims = True ) # Store the means ( needed by the Morpher )
      self.Dc -= self.DcMeans # Columnwise centering
      self.nC = self.Dc.shape[1] # store the updated number of regressors before Morphing
      self.INT_TYPE = HF.FindMinInt( self.Dc.shape[1] ) # find size the index must at least be to account for all columns

    else: self.nC = 0 # no candidate regs if no Dc passed


    # ~~~~~~~~~~~~ Ds Stuff

    if ( self.Ds is None ):
      if ( self.y is not None ): self.Ds = tor.empty( ( len( y ), 0 ) ) # create empty matrix, avoids some if/elses
      else:                      self.Ds = tor.empty( ( 0, 0 ) ) # No information on shape available, will be overwritten by load
      self.DsNames = np.empty( ( 0, ) )  # simplifies the code
    
    else: # A Ds is passed
      if ( tor.isnan( tor.sum( self.Ds ) ) ):         raise AssertionError( "Your Regressor Matrix Ds contains NaNs, Bruh. We don't like that here" )
      if ( len( self.DsNames ) != self.Ds.shape[1] ): raise TypeError( "DsNames has not the same number of elements (columns) as Ds" )
      self.Ds -= tor.mean( Ds, axis = 0, keepdims = True ) # Columnwise centering
      self.Ds, self.DsNames = HF.RemoveDuplicates( self.Ds, self.DsNames )[:2]
    
    self.nS = self.Ds.shape[1] # number of columns ( zero if above if is true )


    # ~~~~~~~~~~~~ Morphing Dictionary
    if ( self.MorphDict is not None ):
      
      if ( self.Dc is None ): raise AssertionError( "No Dc passed, so no regressors can be morphed. For imposed fitting only, don't pass Dc" )

      if ( "NonLinMap" not in self.MorphDict ): raise AssertionError( "MorphDict is missing the key 'NonLinMap', so no information exists on which regressors should be morphed" )
      self.MorphDict["NonLinMap"] = list( np.array( self.MorphDict["NonLinMap"] )[self.DcFilterIdx] ) # Take allowed morphing term set passed by the user and filter deleted duplicates

      if ( self.Dc is not None ): self.MorphDict["nC"] = self.Dc.shape[1] # store the updated (duplicate filtering) number of regressors before Morphing
      else:                       self.MorphDict["nC"] = 0 # no candidate regs if no Dc passed


    # ~~~~~~~~~~~~ Other Stuff
    if ( self.ValFunc is None ): self.ValFunc = lambda theta, L, ERR, Validation, MorphDic, DcFilterIdx : 1 - tor.sum( tor.tensor( ERR ) ) # default to explained variance if no validation function is passed
    
    if ( U is not None ):
      if ( len( U ) <= self.MaxDepth ): raise ValueError( "U must contain at least MaxDepth + 1 elements for the arborescence to have MaxDepth Levels" )
      # TODO: Update all U indices to take into consideration the Dc duplicate filtering
      self.U = U # Take unused index set passed by the user

    else: self.U = [ j for j in range( self.nC ) ] # Nothing was passed, so assume entire dictionary (filtered of duplicates) can be used

  
    # ------------------------------------------------------------------------------------------------ Internal Data ---------------------------------------------------------------------------------
    # the following variables are only used if Dc is not None

    self.Q = Queue.Queue()
    self.LG = MultiKeyHashTable.MultiKeyHashTable()

    self.Abort = False # Toggle activated at MaxDepth to finish regressions early

    self.MinLen = [] # overwritten by first iteration
    self.nTermsNext = 0 # Nodes in next level, overwritten by first iteration
    self.nTermsCurrent = 0 # Nodes in current level, overwritten by first iteration
    self.nComputed = 0 # Nodes computed in current level, stores the current tqdm state

    # For stats only ( not needed by the arbo):
    self.TotalNodes = 1 # Nodes in the arbo ( starts at 1 since root counts )
    self.NotSkipped = 0 # Actually computed regressions ( =0 is used to detect if fit is called for the first time or not )
    self.AbortedRegs = 0 # Not OOIT - computed regressions, since longer than shortest known

    # ------------------------------------------------------------------------------------------------ Results ---------------------------------------------------------------------------------
    # Data of the regressor selected by the validation
    self.theta = None # theta is None is used as verification if the validation function has been done, so no-touchy. Internally a tor.Tensor but returned to the user as ndarray
    self.ERR = None # nd-array
    self.L = None # nd-array


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
      dill.dump( self.DcFilterIdx, file );    ProgressCount.update()
      dill.dump( self.DsNames, file );        ProgressCount.update()
      dill.dump( self.tolRoot, file );        ProgressCount.update()
      dill.dump( self.tolRest, file );        ProgressCount.update()
      dill.dump( self.MaxDepth, file );       ProgressCount.update()
      dill.dump( self.ValFunc, file );        ProgressCount.update()
      dill.dump( self.ValData, file );        ProgressCount.update()
      dill.dump( self.Verbose, file );        ProgressCount.update()
      dill.dump( self.MorphDict, file );      ProgressCount.update()
      dill.dump( self.U, file );              ProgressCount.update()
      dill.dump( self.FileName, file );       ProgressCount.update()
      dill.dump( self.SaveFrequency, file );  ProgressCount.update()
      dill.dump( self.INT_TYPE, file );       ProgressCount.update()

      # Processing data
      dill.dump( self.nC, file );             ProgressCount.update()
      dill.dump( self.nS, file );             ProgressCount.update()
      dill.dump( self.Q, file );              ProgressCount.update()
      dill.dump( self.LG, file );             ProgressCount.update()
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
    ''' Loads the Arbo from a file pickled by the dill module into the calling Arbo object to continue the traversal
    
    ### Inputs:
    - `FileName`: (str) FileName to the pickle file
    - `Print`: (bool) Print some stats
    '''

    with open( FileName, 'rb' ) as file:

      # User Inputs:
      self.y = dill.load( file )
      self.Ds = dill.load( file )
      self.DcMeans = dill.load( file )
      self.Dc = dill.load( file )
      self.DcNames = dill.load( file )
      self.DcFilterIdx = dill.load( file )
      self.DsNames = dill.load( file )
      self.tolRoot = dill.load( file )
      self.tolRest = dill.load( file )
      self.MaxDepth = dill.load( file )
      self.ValFunc = dill.load( file )
      self.ValData = dill.load( file )
      self.Verbose = dill.load( file )
      self.MorphDict = dill.load( file )
      self.U = dill.load( file )
      self.FileName = dill.load( file )
      self.SaveFrequency = dill.load( file )
      self.INT_TYPE = dill.load( file )

      # Processing Data
      self.nC = dill.load( file )
      self.nS = dill.load( file )
      self.Q = dill.load( file )
      self.LG = dill.load( file )
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
        print( "\nTolerance: " + str(self.tolRest), "\nArboDepth:", self.MaxDepth, "\nCurrent Dict-shape:", self.Dc.shape,
              "\nTraversed Nodes:", self.TotalNodes, "\nNot Skipped:", self.NotSkipped, "\nAborted Regs:", self.AbortedRegs )
        print( "\nShortest Sequence at each level:")
        for i in range( len( self.Q.peek() ) ): print( f"Level { i }: { self.MinLen[i] }" ) 
        print() # print empty line

  # ************************************************************************************** rFOrLSR **************************************************************************************
  def rFOrLSR( self, y, Ds = None, Dc = None, U = None, tol = 0.001, MaxTerms = tor.inf, OutputAll = False, LI = None ):
    '''
    Recursive Forward Orthogonal Regression function with imposed regressors capabilities expecting the regressors to be computed in advance and h-stacked in a regressor torch.Tensor.
    The Ds ( pre-selected dictionary ) regressors (if present) are automatically accepted and orthogonalized in appearance order, then regressors selection is performed on Dc's columns
    If the Regressor matrix Dc is None then the algorithm just uses Ds (very quick).
    The function assumes that the regressors and y[k] are centered vectors ( assured by the class CTor ), else the results can be very poor.
    
    ### Inputs: (Most of them taken directly from the class)
    - `y`: ( (p,)-Tensor ) Observed Signal ( Important: assumed mean-free )
    - `Ds`: ( None or ( p, nS )-Tensor ) Selected regressor matrix where each column corresponds to a ( pre- ) selected regressor. None is equivalent to an empty matrix
    - `Dc`: ( None or p, nC )-Tensor ) Candidate regressor matrix where each column corresponds to a candidate regressor
    - `U`: ( None or list of ints ) containing the column numbers of Dc the rFOrLSR is allowed to use, None defaults to all
    - `tol`: ( float ) maximum summed ERR threshold
    - `MaxTerms`: ( float ) determining the maximum number of regressors to be contained in the regression before aborting
    - `OutputAll`: ( bool ) determining if Every Data structure is to be output
    - `LI`: ( None or list of ints ) containing the imposed column numbers of Dc for OOIT-based prediction, not used by the fitting procedure

    Data taken from the Class:
    - `MDict`: ( Dictionary or None ) containing all the required morphing data or None if no morphing is desired
    - `Abort`: ( bool ) if the regression should be aborted if the number MaxTerms is exceeded
    - `Verbose`: ( bool ) creates a regressor counting bar (no progressbar since final amount unknown)
    → Dc can't be taken from the class since that would take away the possibility for validate to call with Dc = None
    → Ds also since it is modified by the Arbo
    
    ### Outputs:
    → The number of outputs depends on what made the regression exit, in most cases only L is returned in a tuple
    - `L`: ( (nr,)-array of self.INT_TYPE ) containing the used regressors' indices (DOESN'T CONTAIN Ds and LI )
    - `theta`: ( (nr,)-array of float ) containing the regression coefficients, empty array if regression aborted
    - `ERR`: ( (nr,)-array of float ) containing the Error reduction ratios of all selected regressors ( Ds then Dc ), empty array if regression aborted
    '''

    # ------------------------------------------------------------------------------------ 0. Init --------------------------------------------------------------------------------
    MatSize = 0 # Solution matrix shape
    # all declared here since part of the output
    if ( self.Verbose ): ProgressCount = tqdm.tqdm( desc = "Current Regression", leave = False, unit = " Regressors" ) # define measuring progress bar
    
    if ( Dc is not None ): # a dictionary serach must be performed ( not only imposed regressors )
      if ( U is None ): U = [j for j in range( Dc.shape[1] )] # set of unused Dc indices ( elements in the set Dc )
      MatSize += len( U ) # can't use Dc.shape since not all regressors might be allowed ( due to Arbo/morphing )
    
    if ( isinstance(  Ds, tor.Tensor ) and Ds.shape == ( len( y ), 0 ) ): Ds = None # overwrite empty array as None
    if ( Ds is not None ):
      if ( Ds.shape == ( len( y ), ) ): Ds = Ds.reshape(-1, 1) # transform to single column
      MatSize += Ds.shape[1] # there are imposed regressors
      if ( MaxTerms < Ds.shape[1] ): raise ValueError( f"MaxTerms ({ MaxTerms }) is smaller than imposed regressors Ds.shape[1] ({ Ds.shape[1] })")

    if ( MatSize == 0 ): raise AssertionError( "Dc and Ds are None, thus no regressors were passed for the algorithm to process" )
    
    L = [] # set of used regressor indices, needed for the output, even if Dc is None
    A = [] # Custom sparse martix for unitriangular matrices, storing only the upper diagonal elements in a list
    s2y = (y @ y).item() # mean free observation empiric variance
    ERR = [] # list of Error reduction ratios of all selected regressors ( Dc and Ds )
    s = 1 # iteration/column count ( 1 since zero-based but only used at 1nd iteration = loop )
    W = [] # Orthogonalized regressors coefficients
    Psi = tor.empty( ( len( y ), 0 ) ); Psi_n = tor.empty( ( len( y ), 0 ) ) # Create empty ( p,0 )-sized matrices to simplify the code below
    
    MOut = None # for the verification in case it's not overwritten by the morphing

    #  ----------------------------------------------------------------------- 1. Imposed regressors orthogonalization ------------------------------------------------------------------------
    
    if ( Ds is not None ): # A matrix was passed as Ds, thus there are pre-selected regressors which are taken in order
      # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
      Psi = Ds[:, 0, None] # unnormed orthogonal regressor matrix ( already centered ) reshaped as column
      n_Omega = HF.Norm2( Psi ) # squared euclidean norm of Omega or fudge factor
      Psi_n = Psi / n_Omega # normed orthogonal regressor matrix
      W.append( ( Psi_n.T @ y ).item() ) # store orthogonal regression coefficient ( w )
      ERR.append( W[-1]**2 * n_Omega / s2y )
      if ( self.Verbose ): ProgressCount.update() # Update counter
      
      for col in range( 1, Ds.shape[1] ): # iterate over columns, start after position 1
        # Computations
        Omega = Ds[:, col] - Psi_n @ ( Psi.T @ Ds[:, col] ) # orthogonalize only the current column ( no reshape needed )
        n_Omega = HF.Norm2( Omega )  # squared euclidean norm of Omega or fudge factor
        W.append( ( Omega @ y ).item() / n_Omega ) # store orthogonal regression coefficient ( w )
        ERR.append( W[-1]**2 * n_Omega / s2y )
        
        # Data storage: add current regressor
        A.append( Psi_n.T @ Ds[:, col, None] ) # Multiply all normed orthogonalized regressors with the currently chosen unorthogonalized regressor
        Psi = tor.column_stack( ( Psi, Omega ) ) # unnormed matrix
        Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) ) # normed matrix
        s += 1 # increment the A-matrix column count
        if ( self.Verbose ): ProgressCount.update() # Update counter

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
      W.append( ( Psi_n[:,-1] @ y ).item() ) # add orthogonal regression coefficient
      ERR.append( W[-1]**2 * n_Reg / s2y )
      U.remove( U[ell] ) # update unused indices list
      if ( self.Verbose ): ProgressCount.update() # Update counter

      # ------------------------------------------------------------------------ 3. Optimal regressor search loop ---------------------------------------------------------------------------
      while ( ( 1 - np.sum( ERR ) > tol ) and ( s < MatSize ) ): # while not enough variance explained ( empty lists sum to zero ) and still regressors available ( nS + nC )
        
        if ( ( self.Abort and ( s > MaxTerms ) )): # for leaf nodes only
          return ( tuple( [np.array( L, dtype = self.INT_TYPE ) ]) ) # R[1/4] this returns a valid L, which is however disqualified as solution candidate being already too long
        
        if ( LI is not None ):
          RegIdx = self.LG.SameStart( np.concatenate( ( LI, L ) ) ) # check if regression is to be continued (doesn't contain Ds content, since no index)
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
        ERR.append( W[-1]**2 * n_Reg / s2y ) # store this regressors ERR
        U.remove( U[ell] ) # update unused indices list
        if ( U == [] ): print( "\n\nThe entire dictionary has been used without reaching the desired tolerance!\n\n" ) # no need to exit, since done by 's'
        s += 1 # increment the A-matrix column count
        if ( self.Verbose ): ProgressCount.update() # Update counter
    
    # ------------------------------------------------------------------------ 4. Output generation ---------------------------------------------------------------------------
    if ( self.Verbose ): ProgressCount.close() # close tqdm counter

    if ( ( s > MaxTerms ) or ( OutputAll == False ) ):
      return ( tuple( [ np.array( L, dtype = self.INT_TYPE ) ]) ) # R[3/4] only triggered if MaxTerms overwritten, L → int, since if self.Dc=None then L=[] is assuemd float
    else:
      # L casted to int for the Dc=None case where L=[] is assuemd float. Return only Dc since MorphData and DcNames are overwritten internally
      return ( HF.SolveSystem( A, W ), np.array( L, dtype = self.INT_TYPE ), np.array( ERR ) ) # R[4/4] return regression coeffs theta and used regressor names ( only selected from Dc )



  # *********************************************************************************************** Actual ADMOrLSR Algorithm **********************************************************************************
  def fit( self, FileName = None, SaveFrequency = 0 ):
    '''Breadth-first Search Arborescent rFOrLSR (AOrLSR)
    
    ### Inputs:
    - `FileName`: (None or str) FileName to the pickle file
    - `SaveFrequency`: (int) frequency in seconds of saving the Arborescence content ( default: 0 for no saving)
    
    ### Output:
    (returns the validation/get_Results function)
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( (nr,)-sized string nd-array ) containing the regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed)
    - `DcNames`: The updated regressor names (duplicate-filtered and potentially morphed)
    '''

    if ( self.y is None ): raise AssertionError( "y is None meaning that an uninitialized Arborescence is being loaded" )
    
    if ( ( not isinstance( SaveFrequency, int ) ) and ( not isinstance( SaveFrequency, float ) ) ):
      raise ValueError( "SaveFrequency must be an integer or a float" )
    
    if ( SaveFrequency < 0 ): raise ValueError( "SaveFrequency cannot be negative" )
    else:                     self.SaveFrequency = SaveFrequency * 60 # overwrite if given, munlt by 60 to make it minutes

    if ( FileName is not None ):
      if ( not isinstance( FileName, str ) ): raise ValueError( "FileName must be a string containing the pickle file" )  
      self.FileName = FileName # overwrite if given

    # --------------------------------------------------------------------------------- Traversal Init ----------------------------------------------------------------------------
    # Root Computation
    
    if ( self.Q.is_empty() and ( self.NotSkipped == 0 ) ): # only true if Arbo non-initialized (or if someone loads an ended arbo from a file, but what's the point?)
      print( "Performing root regression" )
      theta, L, ERR = self.rFOrLSR( self.y, self.Ds, self.Dc, self.U.copy(), self.tolRoot, OutputAll = True ) # create new temp data, use classe's Dc

      self.NotSkipped = 1 # Root is always computed

      self.LG.AddData( np.array( L, dtype = self.INT_TYPE ) ) # Store regression idices
      self.LG.CreateKeys( MinLen = 0, IndexSet = L, Value = 0 ) # declare root subsets for fast lookup
      
      self.MinLen = [ len( L ), len( L ) ] # list so that the MinLen per level is stored, twice since once for root then for first level
      self.MaxDepth = min( ( self.MaxDepth, self.MinLen[-1] ) ) # update to implement ADTT
      self.nTermsNext = 0 # number of nodes in the next level, zero since unknown here
      print( "Shortest encountered sequence (root):", self.MinLen[-1] + self.nS, '\n' )

      for ImposedReg in L: self.Q.put( np.array( [ ImposedReg ], dtype = self.INT_TYPE ) ) # append all imposed regressors to the Queue as one element arrays
      self.nTermsCurrent = len( L ) # update current level number of nodes

      if ( ( self.MaxDepth == 0 ) or ( self.Dc is None ) ): # theta, L, ERR are written by the validation function
        self.MinLen = [ self.MinLen[0] ] # cut out the predictions for the next level, recast into list
        self.Q.clear() # clear the Queue, such that the while is never entered and the arbo goes directly to the validation

    if ( self.SaveFrequency > 0 ): self.MemoryDump( 0 ) # If this first back-up fails we know it immediately, rather than potentially a few hours later
    
    ProgressBar = tqdm.tqdm( total = self.nTermsCurrent, desc = f"Arborescence Level { len( self.Q.peek() ) }", unit = " rFOrLSR" )  # Initialise progrssbar with known total number of iterations of first level 
    ProgressBar.update( self.nComputed ) # insert the correct progress
    StartTime = timeit.default_timer() # start time counter for memory dumps

    # --------------------------------------------------------------------------------- Arborescence Traversal ----------------------------------------------------------------------------
    while ( not self.Q.is_empty() ):
      LI = self.Q.get() # retrieve the next imposed index-set

      if ( ( self.SaveFrequency > 0 ) and ( timeit.default_timer() - StartTime > self.SaveFrequency ) ):
        self.MemoryDump( ProgressBar.n ) # Backup
        StartTime = timeit.default_timer() # reinitialize counter here to not count the backup time
      
      if ( len( LI ) > self.MinLen[-1] ): break # no self.nS is used since also ignored in self.MinLen
      self.TotalNodes += 1 # total number of traversed nodes

      ellG = self.LG.SameStart( LI ) # Regression already computed? Yes = index, No = []

      if ( ellG == [] ): # This indexset isn’t equal to a previous one, else pass and use returned index
        self.NotSkipped += 1 # just for statistics not needed for regression

        U_tmp = self.U.copy() # copy required since else self.U is modified when passed to a function
        for li in LI: 
          if ( li < self.nC ): U_tmp.remove( li ) # remove imposed indices, only not morphed regressors which are not present anyways ( they are however imposed, see above )

        Output = self.rFOrLSR( self.y, tor.column_stack( ( self.Ds, self.Dc[:, LI.astype( np.int64 )] ) ), self.Dc, U_tmp, self.tolRest, MaxTerms = self.MinLen[-1] + self.nS, LI = LI ) # rFOrLSR with Li imposed
        # ellG doesn't contain LI if retrieved from an OOIT prediction, which is however not a problem

        ellG = np.concatenate( ( np.sort( LI ), Output[0] ) ) # sort the current level first indices ( len( LI) )
        if ( len( Output ) == 1 ): Value = self.LG.AddData( np.array( ellG, dtype = self.INT_TYPE ) ) # unknown regression ( not OOIT-predicted): Store in LG and get index
        else:                      Value = Output[1] # regression was OOIT-predicted
        
        self.LG.CreateKeys( MinLen = len( LI ), IndexSet = ellG, Value = Value ) # Declare all regression-starts to LG
        
        self.MinLen[-1] = min( ( len( ellG ), self.MinLen[-1] ) ) # update shortest known reg. sequence ( ellG = LI + L here )
        self.MaxDepth = min( ( self.MaxDepth, self.MinLen[-1] ) ) # update for ADTT
      
      else: ellG = self.LG[ ellG ] # self.LG.SameStart returned an index, so a match was found: retrieve it

    
      if ( len( LI ) < self.MaxDepth ): # Iterate next level? (new regressions are only added by the previous level: if True, AOrLSR is in the last level → don't add nodes for the next one)
        LC = HF.AllCombinations( LI, ellG, self.INT_TYPE ) # compute all term combinations, pass INT_TYPE since not a class member function
        self.nTermsNext += LC.shape[0] # add set cardinality to nTerms of the next level
        for combinations in LC: self.Q.put( combinations ) # append all imposed regressors to the Queue
      
      ProgressBar.update() # increase count, Done here since almost all processing is done for this node, must be before the "last node?" if handling the level change
      
      if ( len( LI ) < len( self.Q.peek() ) ): # reached last node in level?
        ProgressBar.close() # close previous one to avoid glitches before creating new one below ( if not next level )
        print( "Shortest encountered sequence:", self.MinLen[-1] + self.nS, '\n' )

        # Upon new level, keys of current lenght are no longer needed, since LI is longer and thus no requests of that length will ever be made again
        self.LG.DeleteAllOfSize( len( LI ) )
        
        if ( len( LI ) + 1 <= self.MaxDepth ): # Is next level to be iterated?
          ProgressBar = tqdm.tqdm( total = self.nTermsNext, desc = f"Arborescence Level {len( LI ) + 1}", unit = " rFOrLSR" ) # Initialise progressbar with number of iterations
          self.nTermsCurrent = self.nTermsNext # number of node in this level (needed to load the progressbar from file)
          self.MinLen.append( self.MinLen[-1] ) # new entry for the new level

        self.Abort = len( LI ) + 1 == self.MaxDepth # early exit non-competitive regressions since this level contains the leaves ( only true at maxdepth )
        
        self.nTermsNext = 0 # reinitialize next level nodes counter ( must be done after the use for the progressbar )
    
    ProgressBar.close() # Avoids glitches
    
    print( "Finished Arborescence traversal.\nShortest encountered sequence:", self.MinLen[-1] + self.nS, '\n' )
    
    if ( self.SaveFrequency > 0 ): self.MemoryDump( ProgressBar.n ) 
    return ( self.validate() ) # call validation member function itself calling the user vaildation function

  # ********************************************************************************** Best Model selection / Validation **********************************************************************************
  def validate( self ):
    ''' "least regressors = selected" heuristics from iFOrLSR paper wich adds a selection based on the minimal MAE.
    Note: This function can be used to get intermediate results during an arborescence traversal since the search data is not overwritten.

    ### Outputs:
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( (nr,)-sized string nd-array ) containing the regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed)
    - `DcNames`: The updated regressor names (duplicate-filtered and potentially morphed)
    
    '''
    print( "Starting Validation procedure." )
    MinError = tor.inf; Processed = [] # Variables storing the shortest sequence with Min Validation ’s position 

    for reg in tqdm.tqdm( self.LG.Data, desc = "Validating", unit = " Regressions" ): # and over each regressor sequence
      if ( len( reg ) == self.MinLen[-1] ): # compute error metric only if candidate to best regression.
        reg = np.sort( reg ) # sort regressor sequence
        if ( tuple( reg ) not in Processed ): # check that not somehow a permutation of already computed regression
          Processed.append( tuple( reg ) ) # add to processed list, cast to tuple since ndarrays can't use "in"
          
          if ( self.Dc is not None ):
            theta, _, ERR, = self.rFOrLSR( self.y, tor.column_stack( ( self.Ds, self.Dc[:, reg.astype( np.int64 )] ) ), None, None, None, OutputAll = True )
          else: # for regression with imposed regressors only (no Dc)
            theta, _, ERR, = self.rFOrLSR( self.y, self.Ds, None, None, None, OutputAll = True )
          
          if ( self.ValData is None ): Error = 1 - np.sum( ERR ) # take ERR if no validation dictionary is passed
          else:                        Error = self.ValFunc( theta, reg, ERR, self.ValData, self.MorphDict, self.DcFilterIdx ) # compute the passed custom error metric

          if ( Error < MinError ): MinError = Error; self.theta = theta; self.L = reg.astype( np.int64 ); self.ERR = ERR # update best model

    print( f"\nValidation done on { len( Processed ) } different Regressions. Best validation error: { MinError }" )
    
    print( f"Out of { self.TotalNodes } only { self.NotSkipped} regressions were computed, of which { self.AbortedRegs } were OOIT - aborted.\n" )

    return ( self.get_Results() )


  # *************************************************************************** Helper function displaying the regression results ***********************************************************************
  def get_Results( self ):
    ''' Returns the Arborescence results.
    
    ### Outputs:
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( ( nr, )-sized int64 nd-arary ) containing the selected regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: (dict) The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed)
    - `DcNames`: The updated regressor names (duplicate-filtered and potentially morphed)
    '''

    if ( self.theta is None ): raise AssertionError( "No regression results available, thus the fitting hasn't been finished. To get (intermediate) results, use Arbo.validate()" )
    return ( self.theta.cpu().numpy(), self.L, self.ERR, self.MorphDict, self.Dc, self.DcNames )
  

  # *************************************************************************** Helper function changing the Arbo depth if possible ***********************************************************************
  def set_ArboDepth( self, Depth ):
    '''Performs verifications before deciding if changing the Arbo depth is a good plan.'''
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
    ''' Function displaying the regression results in form of two plots (1. Signal comparison, 2. Error distribution) and printing the regressors and their coefficients.
    The second plot (ERR vs MEA) is slightly meaning-less since the ERR and the MAE reduction of each term depends on their position in the BVS.
    Also their order in the BVS is arbitrary since imposed and sorted by the AOrLSR but whatever.
    
    ### Outputs:
    - `FigTuple`: (2D Tuple of Figure objects) for both plots
    - `AxTuple`: (2D Tuple of Axes objects) for both plots
    '''

    if ( self.L is None ): raise AssertionError( "The fitting hasn't been finished, thus no regression results are available. To get intermediate results, trigger the validation procedure" )
    
    if ( self.Dc is not None ): # results in the problem of self.L.shape == (0,)
      RegNames = np.concatenate( (self.DsNames, np.ravel( self.DcNames[self.L] ) ) )
      yHat = tor.column_stack( ( self.Ds, self.Dc[:, self.L] ) ) @ self.theta # Model prediction on the training data
    else: # only a Ds exists
      RegNames = self.DsNames
      yHat = self.Ds @ self.theta # Model prediction on the training data
    
    yNorm = tor.max( tor.abs( self.y ) ) # Compute Model, norming factor to keep the display in % the the error
    Error = ( self.y - yHat ) / yNorm # divide by max abs to norm with the max amplitude
    
    # Metrics
    AbsError = tor.abs( Error )
    MedianAbsDeviation = 100 * tor.median( tor.abs( Error - tor.median( AbsError ) ) ).item() # Outlier stable ( median + abs instead of mean + ()² ) variance-type of metric
    MeanAbsErrorPercent = 100 * tor.mean( AbsError ).item()
    MaxAbsError = 100 * tor.max( AbsError )

    # String formatting
    MeanAbsErrorStr = '{:.3e}'.format( MeanAbsErrorPercent ) if ( MeanAbsErrorPercent < 0.001 ) else str( round( MeanAbsErrorPercent, 3 ) )
    MaxDeviationStr = '{:.3e}'.format( MaxAbsError ) if ( MaxAbsError < 0.001 ) else str( tor.round( MaxAbsError, decimals = 3 ).item() ) # max is stored in array so cheap
    MedianAbsDerivationStr = '{:.4e}'.format( MedianAbsDeviation ) if ( MedianAbsDeviation < 0.001 ) else str( round( MedianAbsDeviation, 3 ) )
    

    # A) Ground thruth and fitting error plot
    Fig, Ax = plt.subplots( 2, sharex = True )
    Ax[0].plot( self.y.cpu(), "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax[0].plot( yHat.cpu(), "tab:orange", marker = '.', markersize = 5 ) # force default orange
    Ax[0].legend( ["System Output y[k]", "Estilmation $\hat{y}$[k]"] )
    Ax[0].grid( which = 'both', alpha = 0.5 )

    Ax[1].plot( Error.cpu(), "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax[1].set_xlim( [0, len( self.y )] )
    Ax[1].set_title( f"{ len( self.theta ) } Terms yielding MAE: { MeanAbsErrorStr }%. Max dev.: { MaxDeviationStr }%. MAD: { MedianAbsDerivationStr }%" )
    Ax[1].legend( ["$y[k]-\hat{y}[k]$"] )
    Ax[1].grid( which = 'both', alpha = 0.5 )
    Fig.tight_layout() # prevents the plot from clipping ticks

    
    # B) ERR stem plots with MAE reduction
    
    # Arg Sort ERR and impose same order on L
    MAE = [] # list containing the MAE values from the models progressively build up
    Order = np.flip( np.argsort( self.ERR ) )
    ERRtemp = self.ERR[ Order ]; Sortedtheta = self.theta[ list( Order ) ]; RegNames = RegNames[Order] # impose same order on all datastructures
    if ( self.Dc is not None ):
      Ltemp = self.L[ Order ]; # L is an empty array if no Dc is passed
      Imposed = tor.column_stack( ( self.Ds, self.Dc[:, Ltemp] ) ) # invalid indexation if Dc is empty or None
    else: Imposed = self.Ds

    for i in range( 1, Imposed.shape[1] + 1 ): # iterate over the index list of all selected regressors. +1 since the indexing excludes the last term
      theta_TMP = self.rFOrLSR( self.y, Ds = Imposed[:, :i], OutputAll = True )[0] # only impose the regressors to get the new theta
      MAE.append( ( tor.mean( tor.abs( self.y - Imposed[:, :i] @ tor.Tensor( theta_TMP ) ) ) / yNorm ).item() )
    
    Fig2, Ax2 = plt.subplots( 2 ) # 2 because the first Ax object is outputted by the function
    Ax2[0].set_title( f"Top: All { len( ERRtemp ) } ERR in descending oder. Bottom: Model MAE evolution" )
    Ax2[0].stem( ERRtemp, linefmt = "#00aaffff" ) # force slightly lighter blue than default blue for compatibility with dark mode
    plt.setp( Ax2[0].get_xticklabels(), visible = False ) # desactivate the ticks
    Ax2[0].set_ylim( [0, 1.05 * max( ERRtemp )] )
    Ax2[0].grid( axis = 'y', alpha = 0.5 )
    
    Ax2[1].plot( MAE, "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax2[1].set_xticks( np.arange( len( ERRtemp ) ), RegNames, rotation = 45, ha = 'right' ) # setting ticks manually is more flexible as it allows rotation
    Ax2[1].grid( axis = 'x', alpha = 0.5 )
    for i, v in enumerate( MAE ): Ax2[1].text( i, v + 0.1 * max( MAE ), '{:.3e}'.format( v ), ha = "center" ) # print exact values
    Ax2[1].set_ylim( [0, max( MAE ) * 1.3] )
    
    Fig2.tight_layout() # prevents the plot from clipping ticks
    plt.subplots_adjust( hspace = 0.001 )
    
    # print summary to console
    print( Imposed.shape[1], "Terms yielding an Mean absolute Error (MAE) of", MeanAbsErrorStr + "% and a maximal deviation of", MaxDeviationStr +
          "% and a Median Absolute Deviation (MAD) of", MedianAbsDerivationStr )
    
    if ( PrintRegressor ): # print the regressors in a readable manner
      print( "\nRecognized regressors:" )
      for i in range( len( Sortedtheta ) ): print( Sortedtheta[i].item(), RegNames[i] )
      print( "\n" )

    return ( ( Fig, Fig2 ), ( Ax, Ax2 ) ) # for further operations