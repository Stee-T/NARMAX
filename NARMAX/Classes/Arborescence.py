import numpy as np
import torch as tor
import matplotlib.pyplot as plt

import dill # memory dumps
import tqdm # progress bars
import timeit # time measurements for Back-ups

# Import from subfolder: ".Name" is subfolder import

# Import from current folder:
from . import Queue
from . import MultiKeyHashTable
from .SymbolicOscillator_0_3 import SymbolicOscillator # for Plot & Print
from . import Morpher

# Import from parent folder:
from .. import HelperFuncs as HF
from ..Validation import InitAndComputeBuffer

# Typing
from typing import Optional, Tuple, Sequence, Union, Callable, Any
from numpy.typing import NDArray

# #################################################################################### Arbo Class ###################################################################################
class Arborescence:
  # ##################################################################################### Init #####################################################################################
  
  def __init__( self, y: Optional[ tor.Tensor ] = None, # System Response / Data
                Ds: Optional[ tor.Tensor ] = None, DsNames: Optional[ NDArray[ np.str_ ] ] = None, # User Imposed Regressors: Dictionary of selected regressors
                Dc: Optional[ tor.Tensor ] = None, DcNames: Optional[ NDArray[ np.str_ ] ] = None, # Fitting Data: dictionary of candidates
                tolRoot: Optional[ float ] = 0.001, tolRest: Optional[ float ] = 0.001, MaxDepth: Optional[ int ] = 5, # Arbo size influencers: rho 1 & 2 and max number of levels
                ValFunc: Optional[ Callable[ [ NDArray[ np.float64 ], NDArray[ np.int64 ], NDArray[ np.float64 ], NDArray[ np.str_ ], dict[ str, Any ], Optional[ NDArray[ np.int64 ] ]], np.float64 ] ] = None,
                ValData: Optional[ dict[str, Any] ] = None, # Validation Data
                Verbose: Optional[ bool ] = False, # rFOrLSR progess feedback
                MorphDict: Optional[ dict[str, Any] ] = None, U: Optional[ Union[ NDArray[ np.int64 ], Sequence[ int ] ] ] = None, # Morphing related stuff
                FileName: Optional[ str ] = None, SaveFrequency: Union[ float, int ] = 0 # Arbo Backup Parameters
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
    - `SaveFrequency`: (float/int) containing the frequency in minutes at which to save the backups (10 = every 10 minutes)

    ### MorphDict content:
    - `NonLinMap`: (list of int) containing the non-linearity indices of all Dc columns allowed to be morphed (0 for no-morphing)
    ### TODO
    '''

    # --------------------------------------------------------------------------------- Type checks --------------------------------------------------------------------------------

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

    if ( ( not isinstance( SaveFrequency, float ) ) and ( not isinstance( SaveFrequency, int ) ) ):
      raise ValueError( "SaveFrequency must be None a float or an int" )


    # --------------------------------------------------------------------------- Store Passed Arguments ---------------------------------------------------------------------------
    # Copy everything into the class

    if ( y is not None ): # Only when initializing empty arbo when loading from a file
      self.y: Optional[ tor.Tensor ] = tor.ravel( y - tor.mean( y ) ) # flatski to (p,) vector for rFOrLSR
      if ( tor.isnan( tor.sum( self.y ) ) ): raise AssertionError( "Your system output y somehow yields NaNs, Bruh. We don't like that here" )
    else: self.y: Optional[ tor.Tensor ] = None

    self.Ds: Optional[ tor.Tensor ] = Ds
    self.Dc: Optional[ tor.Tensor ] = Dc # columnwise centering and duplicate removal performed below
    self.DcMeans: Optional[ NDArray[ np.float64 ] ] = None # Overwritten below, when Dc's non-redundant size (nC) is known after filtering duplicate regressors
    self.DcNames: Optional[ NDArray[ np.str_ ] ] = DcNames
    self.DcFilterIdx: Optional[ NDArray[ np.int64 ] ] = None # indexset of all non-duplicate regressors in Dc, computed below
    self.DsNames: Optional[ NDArray[ np.str_ ] ] = DsNames
    self.tolRoot: Optional[ float ] = tolRoot
    self.tolRest: Optional[ float ] = tolRest
    self.MaxDepth: Optional[ int ] = MaxDepth if MaxDepth is not None else 5
    self.ValFunc: Optional[ Callable[ [ NDArray[ np.float64 ], NDArray[ np.int64 ], NDArray[ np.float64 ], NDArray[ np.str_ ], dict[ str, Any ], Optional[ NDArray[ np.int64 ] ]], np.float64 ] ] = ValFunc
    self.ValData: Optional[ dict[ str, Any ] ] = ValData
    self.Verbose: Optional[ bool ] = Verbose
    self.MorphDict: Optional[ dict[ str, Any ] ] = MorphDict
    self.FileName: Optional[ str ] = FileName
    self.SaveFrequency: Optional[ Union[ float, int ] ] = SaveFrequency * 60 # transform from minutes into seconds
    self.INT_TYPE = np.int64 # default, if no DC present to overwrite it (thus not used since it only affects the arbo search)

    # ----------------------------------------------------------------------------- Argument Processing ----------------------------------------------------------------------------
    # ~~~~~~~~~~~~ DC Stuff
    if ( self.DcNames is not None ): # needs to be checked first since used in Dc processing -if below
      if ( len( self.DcNames ) != self.Dc.shape[1] ): raise TypeError( "DcNames must be None or a np.array of the same length as Dc" )

    if ( self.Dc is not None ):
      if ( tor.isnan( tor.sum( self.Dc ) ) ): raise AssertionError( "Regressor Matrix Dc contains NaNs, Bruh. We don't like that here" )

      self.Dc, self.DcNames, self.DcFilterIdx = HF.RemoveDuplicates( self.Dc, self.DcNames )
      self.DcMeans = tor.mean( self.Dc, axis = 0, keepdims = True ) # Store the means ( needed by the Morpher )
      self.Dc -= self.DcMeans # Columnwise centering
      self.nC: int = self.Dc.shape[1] # store the updated number of regressors before Morphing

      # Determe the numpy integer dtype necessary to hold the current nCols to reduce memory usage
      if   ( self.Dc.shape[1] <= np.iinfo( np.uint16 ).max ): self.INT_TYPE = np.uint16
      elif ( self.Dc.shape[1] <= np.iinfo( np.uint32 ).max ): self.INT_TYPE = np.uint32 # certainly sufficient in most cases
      else: self.INT_TYPE = np.uint64

    else: self.nC = 0 # no candidate regs if no Dc passed


    # ~~~~~~~~~~~~ Ds Stuff

    if ( self.Ds is None ):
      if ( self.y is not None ): self.Ds = tor.empty( ( len( y ), 0 ) ) # create empty matrix, avoids some if/elses
      else:                      self.Ds = tor.empty( ( 0, 0 ) ) # No information on shape available, will be overwritten by load()
      self.DsNames = np.empty( ( 0, ) )  # simplifies the code
    
    else: # A Ds is passed
      if ( tor.isnan( tor.sum( self.Ds ) ) ):         raise AssertionError( "Your Regressor Matrix Ds contains NaNs. We don't like that here" )
      if ( len( self.DsNames ) != self.Ds.shape[1] ): raise TypeError( "DsNames has not the same number of elements (columns) as Ds" )
      self.Ds -= tor.mean( Ds, axis = 0, keepdims = True ) # Columnwise centering
      self.Ds, self.DsNames = HF.RemoveDuplicates( self.Ds, self.DsNames )[:2]
    
    self.nS: int = self.Ds.shape[1] # number of columns ( zero if above if is true )


    # ~~~~~~~~~~~~ Morphing Dictionary
    if ( self.MorphDict is not None ):
      
      if ( self.Dc is None ): raise AssertionError( "No Dc passed, so no regressors can be morphed. For imposed fitting only, don't pass Dc" )

      if ( "NonLinMap" not in self.MorphDict ): raise AssertionError( "MorphDict is missing the key 'NonLinMap', so no information exists on which regressors should be morphed" )
      self.MorphDict["NonLinMap"] = list( np.array( self.MorphDict[ "NonLinMap" ] )[ self.DcFilterIdx ] ) # Take allowed morphing term set passed by the user and filter deleted duplicates

      if ( self.Dc is not None ): self.MorphDict["nC"] = self.Dc.shape[1] # store the updated (duplicate filtering) number of regressors before Morphing
      else:                       self.MorphDict["nC"] = 0 # no candidate regs if no Dc passed


    # ~~~~~~~~~~~~ Other Stuff
    # TODO: This shuold probably use the default validation instead
    if ( self.ValFunc is None ): # default to explained variance if no validation function is passed
      self.ValFunc = lambda theta, L, ERR, RegNames, ValidationData: 1 - tor.sum( tor.tensor( ERR ) ) 
    
    if ( U is not None ):
      if ( len( U ) <= self.MaxDepth ): raise ValueError( "U must contain at least MaxDepth + 1 elements for the arborescence to have MaxDepth Levels" )
      # TODO: Update all U indices to take into consideration the Dc duplicate filtering
      self.U: Union[ Sequence[ int ], NDArray[ np.int64 ] ] = U # Take unused index set passed by the user

    else: self.U: Union[ Sequence[ int ], NDArray[ np.int64 ] ] = [ j for j in range( self.nC ) ] # Nothing was passed, so assume entire dictionary (filtered of duplicates) can be used

  
    # -------------------------------------------------------------------------------- Internal Data -------------------------------------------------------------------------------
    # the following variables are only used if Dc is not None

    self.Q: Queue.Queue = Queue.Queue()
    self.LG: MultiKeyHashTable.MultiKeyHashTable = MultiKeyHashTable.MultiKeyHashTable()

    self.Abort: bool = False # Toggle activated at MaxDepth to finish regressions early

    self.MinLen: list[ int ] = [] # overwritten by first iteration
    self.nNodesInNextLevel: int = 0 # Nodes in next level, overwritten by first iteration
    self.nNodesInCurrentLevel: int = 0 # Nodes in current level, overwritten by first iteration
    self.nComputed: int = 0 # Nodes computed in current level, stores the current tqdm state

    # For stats only ( not needed by the arbo):
    self.TotalNodes: int = 1 # Nodes in the arbo ( starts at 1 since root counts )
    self.nNotSkippedNodes: int = 0 # Actually computed regressions ( =0 is used to detect if fit is called for the first time or not )
    self.AbortedRegs: int = 0 # Not OOIT - computed regressions, since longer than shortest known

    # ----------------------------------------------------------------------------------- Results ----------------------------------------------------------------------------------
    # Data of the regressor selected by the validation
    self.theta: Optional[ tor.Tensor ] = None # theta as None is used as verification if the validation function has been done, so no-touchy. Internally a tor.Tensor but returned to the user as ndarray
    self.ERR: Optional[ NDArray[ np.float64 ] ] = None
    self.L: Optional[ NDArray[ self.INT_TYPE ] ] = None


  ################################################################################### Memory dump ##################################################################################
  def MemoryDump( self, nComputed: int ):
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
      dill.dump( self.nNodesInNextLevel, file );     ProgressCount.update()
      dill.dump( self.nNodesInCurrentLevel, file );  ProgressCount.update()
      dill.dump( nComputed, file );           ProgressCount.update() # from function input, used to display the correct progress
      dill.dump( self.TotalNodes, file );     ProgressCount.update()
      dill.dump( self.nNotSkippedNodes, file );     ProgressCount.update()
      dill.dump( self.AbortedRegs, file );    ProgressCount.update()
    
    ProgressCount.close()


  ################################################################################### Memory Load ##################################################################################
  def load( self, FileName: str, Print: bool = True ): 
    ''' Loads the Arbo from a file pickled by the dill module into the calling Arbo object to continue the traversal
    
    ### Inputs:
    - `FileName`: (str) FileName to the pickle file
    - `Print`: (bool) Print some stats
    '''
    # Note all the below values are strictly speaking optionals, however here they are guaranteed to have a value

    with open( FileName, 'rb' ) as file:

      # User Inputs:
      self.y: Optional[ tor.Tensor ] = dill.load( file )
      self.Ds: Optional[ tor.Tensor ] = dill.load( file )
      self.DcMeans: Optional[ tor.Tensor ] = dill.load( file )
      self.Dc: Optional[ tor.Tensor ] = dill.load( file )
      self.DcNames: Optional[ NDArray[ np.str_ ] ] = dill.load( file )
      self.DcFilterIdx: Optional[ NDArray[ int ] ] = dill.load( file )
      self.DsNames: Optional[ NDArray[ np.str_ ] ] = dill.load( file )
      self.tolRoot: Optional[ float ] = dill.load( file )
      self.tolRest: Optional[ float ] = dill.load( file )
      self.MaxDepth: Optional[ int ] = dill.load( file )
      self.ValFunc: Optional[ Callable[ [ NDArray[ np.float64 ], NDArray[ np.int64 ], NDArray[ np.float64 ], NDArray[ np.str_ ], dict[ str, Any ], Optional[ NDArray[ np.int64 ] ]], np.float64 ] ] = dill.load( file )
      self.ValData: Optional[ dict[ str, Any ] ] = dill.load( file )
      self.Verbose: Optional[ bool ] = dill.load( file )
      self.MorphDict: Optional[ dict[ str, Any ] ] = dill.load( file )
      self.U: Union[ Sequence[ int ], NDArray[ np.int64 ] ] = dill.load( file )
      self.FileName: Optional[ str ] = dill.load( file )
      self.SaveFrequency: Optional[ Union[ float, int ] ] = dill.load( file )
      self.INT_TYPE = dill.load( file )

      # Processing Data
      self.nC: int = dill.load( file )
      self.nS: int = dill.load( file )
      self.Q: Queue.Queue = dill.load( file )
      self.LG: MultiKeyHashTable.MultiKeyHashTable  = dill.load( file )
      self.Abort: bool = dill.load( file )

      # Arbo Statistics
      self.MinLen: list[ int ] = dill.load( file )
      self.nNodesInNextLevel: int = dill.load( file )
      self.nNodesInCurrentLevel: int = dill.load( file )
      self.nComputed: int = dill.load( file )
      self.TotalNodes: int = dill.load( file )
      self.nNotSkippedNodes: int = dill.load( file )
      self.AbortedRegs: int = dill.load( file )

      if ( Print ): # print some stats
        print( "\nTolerance: " + str(self.tolRest), "\nArboDepth:", self.MaxDepth, "\nCurrent Dict-shape:", self.Dc.shape,
              "\nTraversed Nodes:", self.TotalNodes, "\nNot Skipped:", self.nNotSkippedNodes, "\nAborted Regs:", self.AbortedRegs )
        print( "\nShortest Sequence at each level:")
        for i in range( len( self.Q.peek() ) ): print( f"Level { i }: { self.MinLen[i] }" ) 
        print() # print empty line

  ##################################################################################### rFOrLSR ####################################################################################
  def rFOrLSR( self, y: tor.Tensor, Ds: Optional[ tor.Tensor ] = None, Dc: Optional[ tor.Tensor ] = None, U: Optional[ Sequence[ int ] ] = None,
               tol: float = 0.001, MaxTerms: Union[ int, float ] = tor.inf, OutputAll: bool = False, LI: Optional[ Sequence[ int ] ] = None ):
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

    # ----------------------------------------------------------------------------------- 0. Init ----------------------------------------------------------------------------------
    MatSize: int = 0 # Solution matrix shape
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
    
    L: list[ int ] = [] # set of used regressor indices, needed for the output, even if Dc is None
    A: list[ list[ int ] ] = [] # Custom sparse martix for unitriangular matrices, storing only the upper diagonal elements in a list
    s2y: float = ( y @ y ).item() # mean free observation empiric variance
    ERR = [] # list of Error reduction ratios of all selected regressors ( Dc and Ds )
    s: int = 1 # iteration/column count ( 1 since zero-based but only used at 1nd iteration = loop )
    W: list[ float ] = [] # Orthogonalized regressors coefficients
    Psi: tor.Tensor = tor.empty( ( len( y ), 0 ) ); Psi_n: tor.Tensor = tor.empty( ( len( y ), 0 ) ) # Create empty ( p,0 )-sized matrices to simplify the code below
    
    MOut = None # for the verification in case it's not overwritten by the morphing

    # ------------------------------------------------------------------- 1. Imposed regressors orthogonalization ------------------------------------------------------------------
    
    if ( Ds is not None ): # A matrix was passed as Ds, thus there are pre-selected regressors which are taken in order
      # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
      Psi = Ds[:, 0, None] # unnormed orthogonal regressor matrix ( already centered ) reshaped as column
      n_Omega = HF.Norm2( Psi ) # squared euclidean norm of Omega or fudge factor
      Psi_n = Psi / n_Omega # normed orthogonal regressor matrix
      W.append( ( Psi_n.T @ y ).item() ) # store orthogonal regression coefficient ( w )
      ERR.append( W[-1]**2 * n_Omega / s2y )
      if ( self.Verbose ): ProgressCount.update()
      
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
        if ( self.Verbose ): ProgressCount.update()

    # Term selecting iteration with orthogonalization
    if ( Dc is not None ):
      # ----------------------------------------------------- 2. First iteration treated separately since no orthogonalization -----------------------------------------------------
      if ( Ds is not None ): Omega = Dc[:, U] - Psi_n @ ( Psi.T @ Dc[:, U] ) # orthogonalize Dc w.r.t Ds
      else: Omega = Dc[:, U] # If no imposed term start with unorthogonalized dictionary
      
      n_Omega = HF.Norm2( Omega ) # norm squared or fudge factor elementwise
      ell = tor.argmax( ( Omega.T @ y )**2 / tor.ravel( n_Omega ) ) # get highest QERR index

      if ( self.MorphDict is not None ): # Morphing necessary
        MOut = Morpher.DictionaryMorpher( U, ell, Psi, Psi_n, y, A, W, L, Ds, Dc, self.MorphDict )
        
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
      if ( self.Verbose ): ProgressCount.update()

      # --------------------------------------------------------------------- 3. Optimal regressor search loop ---------------------------------------------------------------------
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
          MOut = Morpher.DictionaryMorpher( U, ell, Psi, Psi_n, y, A, W, L, Ds, Dc, self.MorphDict ) # s+1 since those entries are being written in and must thus be included
          
          if ( MOut is not None ): # Function has been morphed
            self.MorphDict["MorphData"].append( MOut[0] )
            self.MorphDict["DcNames"].append( MOut[2] ) # Parse Morphing Output
            L.append( Dc.shape[1] ) # append newly added Regressor
            Dc = tor.column_stack( ( Dc, MOut[1] ) ) # append morphed term to dictionary ( unorthogonalized )
            Reg: tor.Tensor = MOut[1] - Psi_n @ ( Psi.T @ MOut[1] ) # orthogonalize ( no reshape needed )
            n_Reg: Union[ float, tor.Tensor ] = HF.Norm2( Reg )  # squared euclidean norm of Reg or fudge factor

        if ( ( self.MorphDict is None ) or ( MOut is None ) ): # no morphing, since either 1 ) deactivated or 2 ) not morphable
          L.append( U[ell] ) # add the unmorphed = original term to the regression
          Reg: tor.Tensor = Omega[:, ell] # selected regressor
          if ( len( U ) > 1 ): n_Reg = n_Omega[0, ell].item() # selected Regressor norm
          else: n_Reg: Union[ float, tor.Tensor ] = n_Omega # TODO check what's going on # must be an array not a float, holds for length zero and 1
          # else: n_Reg = tor.tensor( [[n_Omega]] ) # must be an array not an int, holds for length zero and 1

        # 2. Data storage
        A.append( Psi_n.T @ Dc[:, L[-1]] ) # Multiply all normed orthogonalized regressors with the currently chosen unorthogonalized regressor
        Psi = tor.column_stack( ( Psi, Reg ) ) # unnormed matrix
        Psi_n = tor.column_stack( ( Psi_n, Reg / n_Reg ) ) # normed matrix
        W.append( ( Psi_n[:, -1] @ y ).item() ) # store orthogonal regression coefficient ( w )
        ERR.append( W[-1]**2 * n_Reg / s2y ) # store this regressors' ERR
        U.remove( U[ell] ) # update unused indices list
        if ( U == [] ): print( "\n\nThe entire dictionary has been used without reaching the desired tolerance!\n\n" ) # no need to exit, since done by 's'
        s += 1 # increment the A-matrix column count
        if ( self.Verbose ): ProgressCount.update()
         
    # ---------------------------------------------------------------------------- 4. Output generation ----------------------------------------------------------------------------
    if ( self.Verbose ): ProgressCount.close() # close tqdm counter

    if ( ( s > MaxTerms ) or ( OutputAll == False ) ):
      return ( tuple( [ np.array( L, dtype = self.INT_TYPE ) ]) ) # R[3/4] only triggered if MaxTerms overwritten, L → int, since if self.Dc=None then L=[] is assuemd float
    else:
      # L casted to int for the Dc=None case where L=[] is assuemd float. Return only Dc since MorphData and DcNames are overwritten internally
      return ( HF.SolveSystem( A, W ), np.array( L, dtype = self.INT_TYPE ), np.array( ERR ) ) # R[4/4] return regression coeffs theta and used regressor names ( only selected from Dc )


  ############################################################################ Actual ADMOrLSR Algorithm ###########################################################################
  def fit( self, FileName: Optional[ str ] = None, SaveFrequency: Union[ int, float ] = 0 ):
    '''Breadth-first Search Arborescent rFOrLSR (AOrLSR)
    
    ### Inputs:
    - `FileName`: (None or str) FileName to the pickle file
    - `SaveFrequency`: (int or float) frequency in seconds of saving the Arborescence content ( default: 0 for no saving)
    
    ### Output:
    (returns the validation/get_Results function)
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( (nr,)-sized string nd-array ) containing the regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed, mean-free)
    - `DcNames`: The updated regressor names (duplicate-filtered and potentially morphed)
    '''

    if ( self.y is None ): raise AssertionError( "y is None meaning that an uninitialized Arborescence is being loaded" )
    
    if ( ( not isinstance( SaveFrequency, int ) ) and ( not isinstance( SaveFrequency, float ) ) ):
      raise ValueError( "SaveFrequency must be an integer or a float" )
    
    if ( SaveFrequency < 0 ): raise ValueError( "SaveFrequency cannot be negative" )
    else:                     self.SaveFrequency = int( SaveFrequency ) * 60 # overwrite if given, munlt by 60 to make it minutes

    if ( FileName is not None ):
      if ( not isinstance( FileName, str ) ): raise ValueError( "FileName must be a string containing the pickle file" )  
      self.FileName = FileName # overwrite if given

    # ------------------------------------------------------------------------------- Traversal Init -------------------------------------------------------------------------------
    # Root Computation
    
    if ( self.Q.is_empty() and ( self.nNotSkippedNodes == 0 ) ): # only true if Arbo non-initialized (or if someone loads an ended arbo from a file, but what's the point?)
      print( "Performing root regression" )
      _, L, _ = self.rFOrLSR( self.y, self.Ds, self.Dc, self.U.copy(), self.tolRoot, OutputAll = True ) # create new temp data, use classe's Dc

      self.nNotSkippedNodes = 1 # Root is always computed

      self.LG.AddData( np.array( L, dtype = self.INT_TYPE ) ) # Store regression idices
      self.LG.CreateKeys( MinLen = 0, IndexSet = L, Value = 0 ) # declare root subsets for fast lookup
      
      self.MinLen = [ len( L ), len( L ) ] # list so that the MinLen per level is stored, twice since once for root then for first level
      self.MaxDepth = min( ( self.MaxDepth, self.MinLen[-1] ) ) # update to implement ADTT
      self.nNodesInNextLevel = 0 # number of nodes in the next level, zero since unknown here
      print( "Shortest encountered sequence (root):", self.MinLen[-1] + self.nS, '\n' )

      for ImposedReg in L: self.Q.put( np.array( [ ImposedReg ], dtype = self.INT_TYPE ) ) # append all imposed regressors to the Queue as one element arrays
      self.nNodesInCurrentLevel = len( L )

      if ( ( self.MaxDepth == 0 ) or ( self.Dc is None ) ): # theta, L, ERR are written by the validation function
        self.MinLen = [ self.MinLen[0] ] # cut out the predictions for the next level, recast into list
        self.Q.clear() # clear the Queue, such that the while is never entered and the arbo goes directly to the validation

    if ( self.SaveFrequency > 0 ): self.MemoryDump( 0 ) # If this first back-up fails we know it immediately, rather than potentially a few hours later
    
    ProgressBar = tqdm.tqdm( total = self.nNodesInCurrentLevel, desc = f"Arborescence Level { len( self.Q.peek() ) }", unit = " rFOrLSR" )  # Initialise progrssbar with known total number of iterations of first level 
    ProgressBar.update( self.nComputed ) # insert the correct progress
    StartTime: float = timeit.default_timer() # start time counter for memory dumps

    # --------------------------------------------------------------------------- Arborescence Traversal ---------------------------------------------------------------------------
    while ( not self.Q.is_empty() ):
      LI: NDArray[ np.int64 ] = self.Q.get() # retrieve the next imposed index-set

      if ( ( self.SaveFrequency > 0 ) and ( timeit.default_timer() - StartTime > self.SaveFrequency ) ):
        self.MemoryDump( ProgressBar.n ) # Backup
        StartTime = timeit.default_timer() # reinitialize counter here to not count the backup time
      
      if ( len( LI ) > self.MinLen[-1] ): break # no self.nS is used since also ignored in self.MinLen
      self.TotalNodes += 1 # total number of traversed nodes

      ellG = self.LG.SameStart( LI ) # Regression already computed? Yes = index, No = []

      if ( ellG == [] ): # This indexset isn’t equal to a previous one, else pass and use returned index
        self.nNotSkippedNodes += 1 # just for statistics not needed for regression

        U_tmp: list[ int ] = self.U.copy() # copy required since else self.U is modified when passed to a function
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
        self.nNodesInNextLevel += LC.shape[0] # add set cardinality to nTerms of the next level
        for combinations in LC: self.Q.put( combinations ) # append all imposed regressors to the Queue
      
      ProgressBar.update() # increase count, Done here since almost all processing is done for this node, must be before the "last node?" if handling the level change
      
      if ( len( LI ) < len( self.Q.peek() ) ): # reached last node in level?
        ProgressBar.close() # close previous one to avoid glitches before creating new one below ( if not next level )
        print( "Shortest encountered sequence:", self.MinLen[-1] + self.nS, '\n' )

        # Upon new level, keys of current lenght are no longer needed, since LI is longer and thus no requests of that length will ever be made again
        self.LG.DeleteAllOfSize( len( LI ) )
        
        if ( len( LI ) + 1 <= self.MaxDepth ): # Is next level to be iterated?
          ProgressBar = tqdm.tqdm( total = self.nNodesInNextLevel, desc = f"Arborescence Level {len( LI ) + 1}", unit = " rFOrLSR" ) # Initialise progressbar with number of iterations
          self.nNodesInCurrentLevel = self.nNodesInNextLevel # number of node in this level (needed to load the progressbar from file)
          self.MinLen.append( self.MinLen[-1] ) # new entry for the new level

        self.Abort = len( LI ) + 1 == self.MaxDepth # early exit non-competitive regressions since this level contains the leaves
        
        self.nNodesInNextLevel = 0 # must be done after the use for the progressbar
    
    ProgressBar.close() # Avoids glitches
    
    print( "Finished Arborescence traversal.\nShortest encountered sequence:", self.MinLen[-1] + self.nS, '\n' )
    
    if ( self.SaveFrequency > 0 ): self.MemoryDump( ProgressBar.n ) 
    return ( self.validate() ) # call validation member function itself calling the user vaildation function

  ######################################################################## Best Model selection / Validation #######################################################################
  def validate( self ) -> Tuple[ NDArray[ np.float64 ], NDArray[ np.int64 ], NDArray[ np.float64 ], dict[str, Any], tor.Tensor, NDArray[ np.str_ ] ]:
    ''' "least regressors = selected" heuristics from iFOrLSR paper wich adds a selection based on the minimal MAE.
    Note: This function can be used to get intermediate results during an arborescence traversal since the search data is not overwritten.

    ### Outputs:
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( (nr,)-sized string nd-array ) containing the regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed, mean-free)
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
            RegNames: NDArray[ np.str_ ] = np.concatenate( ( self.DsNames, np.ravel( self.DcNames[reg] ) ) ) # pass only selected regressors' RegNames
          else: # for regression with imposed regressors only (no Dc)
            theta, _, ERR, = self.rFOrLSR( self.y, self.Ds, None, None, None, OutputAll = True )
            RegNames: NDArray[ np.str_ ] = self.DsNames
          
          # TODO: remove this and let validate default to the default validation function.
          if ( self.ValData is None ): Error: float = 1 - np.sum( ERR ) # take ERR if no validation dictionary is passed
          else:                        Error: float = self.ValFunc( theta, reg, ERR, RegNames, self.ValData, self.DcFilterIdx ) # compute the passed custom error metric

          if ( Error < MinError ): MinError: float = Error; self.theta = theta; self.L = reg.astype( np.int64 ); self.ERR = ERR # update best model

    if ( MinError == tor.inf ):
      print( "\n\nValidation failed: All regressions yield inf as validation error. Outputtng one of minimal length\n\n" )
      for reg in self.LG.Data: # and over each regressor sequence
        if ( len( reg ) == self.MinLen[-1] ): self.theta = theta; self.L = reg.astype( np.int64 ); self.ERR = ERR; break

    print( f"\nValidation done on { len( Processed ) } different Regressions. Best validation error: { MinError }\n",
           f"Out of { self.TotalNodes } only { self.nNotSkippedNodes } regressions were computed, of which { self.AbortedRegs } were OOIT-aborted.\n" )

    return ( self.get_Results() )


  ############################################################################ Regression Results getter ###########################################################################
  def get_Results( self ) -> Tuple[ NDArray[ np.float64 ], NDArray[ np.int64 ], NDArray[ np.float64 ], dict[str, Any], tor.Tensor, NDArray[ np.str_ ] ]:
    ''' Returns the Arborescence results.
    
    ### Outputs:
    - `theta`: ( (nr,)-sized float nd-array ) containing the estimated regression coefficients
    - `L`: ( ( nr,)-sized int64 nd-arary ) containing the selected regressors indices
    - `ERR`: ( (nr,)-sized float nd-array ) containing the regression's ERR
    - `MorphDict`: (dict) The generated morphing dictionary containing morphing meta-data
    - `Dc`: The updated regressor dictionary (duplicate-filtered and potentially morphed, mean-free)
    - `DcNames`: The updated regressor names (duplicate-filtered and potentially morphed)
    '''

    if ( self.theta is None ): raise AssertionError( "No regression results available, thus the fitting hasn't been finished. To get (intermediate) results, use Arbo.validate()" )
    return ( self.theta.cpu().numpy(), self.L, self.ERR, self.MorphDict, self.Dc, self.DcNames )
  

  ############################################################### Helper function changing the Arbo depth if possible ##############################################################
  def set_ArboDepth( self, Depth: int ) -> None:
    '''Performs verifications before deciding if changing the Arbo depth is a good plan.'''
    # The order of the following checks is important to output the real error, as conditions overlap.
    if ( Depth < 0 ):                    raise AssertionError( "Depth can't be negative" )
    if ( Depth < len( self.Q.peek() ) ): raise AssertionError( f"The Arbo has already passed that depth. The current depth is { len( self.Q.peek() ) }" )
    if ( self.Abort ):                   raise AssertionError( "The Arbo is already in its last level. It can't be made deeper since leaf nodes don't produce children" )
    
    if ( Depth > self.MinLen[-1] ):
      print (f"\n\nWarning: Desired depth is greater than the shortest known sequence ({ self.MinLen[-1] }). Thus no update was performed.\n\n" )
      return # prevents overwriting by early exiting the function.
    
    self.MaxDepth = Depth # nothing talks against it, if we arrived until here :D


  ################################################################################## Plot and Print ################################################################################
  def PlotAndPrint( self, ValData: dict[str, Any], PrintRegressor: bool = True ):
    ''' Function displaying the regression results in form of two plots (1. Signal comparison, 2. Error distribution) and printing the regressors and their coefficients.
    The second plot (ERR vs MEA) is slightly meaning-less since the ERR and the MAE reduction of each term depends on their position in the BVS.
    Also their order in the BVS is arbitrary since imposed and sorted by the AOrLSR but whatever.
    
    ### Outputs:
    - `FigTuple`: (2D Tuple of Figure objects) for both plots
    - `AxTuple`: (2D Tuple of Axes objects) for both plots
    '''

    if ( self.L is None ): raise AssertionError( "The fitting hasn't been finished, thus no regression results are available."
                                                 "To get intermediate results, trigger the validation procedure" )

    # ------------------------------------------------------------------------- Figure 1. Signal comparison ------------------------------------------------------------------------

    if ( self.Dc is not None ): RegNames: NDArray[ np.str_ ] = np.concatenate( ( self.DsNames, np.ravel( self.DcNames[ self.L ] ) ) ) # avoids incorrect indexation with L.shape == (0,) and Dc = None
    else:                       RegNames: NDArray[ np.str_ ] = self.DsNames # only a Ds exists
  
    # Initialize the model
    if ( "OutputVarName" not in ValData.keys() ): OutputVarName: str = "y" # default if not passed
    else:                                         OutputVarName: str = ValData["OutputVarName"]

    Model: SymbolicOscillator = SymbolicOscillator( ValData["InputVarNames"], ValData["NonLinearities"], RegNames, self.theta, OutputVarName )
    yHat: tor.Tensor = InitAndComputeBuffer( Model, ValData["y"][0], ValData["Data"][0] )

    yNorm: float = tor.max( tor.abs( ValData["y"][0] ) ).item() # Compute Model, norming factor to keep the display in % the the error
    Error: tor.Tensor = ( ValData["y"][0] - yHat ) / yNorm # divide by max abs to norm with the max amplitude
    
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
    Ax[0].plot( ValData["y"][0].cpu(), "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax[0].plot( yHat.cpu(), "tab:orange", marker = '.', markersize = 5 ) # force default orange
    Ax[0].legend( ["System Output y[k]", "Estilmation $\\hat{y}$[k]"] )
    Ax[0].grid( which = 'both', alpha = 0.5 )

    Ax[1].plot( Error.cpu(), "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax[1].set_xlim( [0, len( ValData["y"][0] )] )
    Ax[1].set_title( f"{ len( self.theta ) } Terms yielding MAE: { MeanAbsErrorStr }%. Max dev.: { MaxDeviationStr }%. MAD: { MedianAbsDerivationStr }%" )
    Ax[1].legend( ["$y[k]-\\hat{y}[k]$"] )
    Ax[1].grid( which = 'both', alpha = 0.5 )
    Fig.tight_layout() # prevents the plot from clipping ticks
    
    # ----------------------------------------------------------------- Figure 2. ERR stem plots with MAE reduction ----------------------------------------------------------------
    # Arg Sort ERR and impose same order on L
    MAE: list[ float ] = [] # list containing the MAE values from the models progressively build up
    Order: NDArray[ np.int64 ] = np.flip( np.argsort( self.ERR ) )
    SortedERR: NDArray[ np.float64 ] = self.ERR[ Order ]; RegNames: NDArray[ np.str_ ] = RegNames[ Order ] # impose same order on all datastructures
    
    if ( self.Dc is not None ): Imposed: tor.Tensor = tor.column_stack( ( self.Ds, self.Dc[:, self.L[ Order ] ] ) ) # invalid indexation if Dc is empty or None
    else:                       Imposed: tor.Tensor = self.Ds

    # The procedure doesn't discriminate between Ds and Dc, since Ds might also contain AR terms which must thus be processed
    for i in tqdm.tqdm( range( 1, SortedERR.shape[0] + 1 ), desc = "MAE: Estimating Sub-Models ", leave = False ):
      theta_TMP = self.rFOrLSR( self.y, Ds = Imposed[:, :i], OutputAll = True )[0] # Estimate Sub-model theta from training data
      Model = SymbolicOscillator( ValData["InputVarNames"], ValData["NonLinearities"], RegNames[:i], theta_TMP, OutputVarName ) # Generate current submodel
      yHat = InitAndComputeBuffer( Model, ValData["y"][0], ValData["Data"][0] )
      MAE.append( ( tor.mean( tor.abs( ValData["y"][0] - yHat ) ) / yNorm ).item() ) # Compute the error

    Fig2, Ax2 = plt.subplots( 2, sharex = True ) # 2 because the first Ax object is outputted by the function
    Ax2[0].set_title( f"Top: All { len( SortedERR ) } ERR in descending oder. Bottom: Model MAE evolution" )
    Ax2[0].stem( SortedERR, linefmt = "#00aaffff" ) # force slightly lighter blue than default blue for compatibility with dark mode
    plt.setp( Ax2[0].get_xticklabels(), visible = False ) # desactivate the ticks
    Ax2[0].set_ylim( [0, 1.05 * max( SortedERR )] )
    Ax2[0].grid( axis = 'y', alpha = 0.5 )
    
    Ax2[1].plot( MAE, "#00aaffff", marker = '.', markersize = 5 ) # force slightly lighter blue than default blue for compatibility with dark mode
    Ax2[1].set_xticks( np.arange( len( SortedERR ) ), RegNames, rotation = 45, ha = 'right' ) # setting ticks manually is more flexible as it allows rotation
    Ax2[1].grid( axis = 'x', alpha = 0.5 )
    for i, v in enumerate( MAE ): Ax2[1].text( i, v + 0.1 * max( MAE ), '{:.3e}'.format( v ), ha = "center" ) # print exact values
    Ax2[1].set_ylim( [0, max( [i if i < np.inf else 0 for i in MAE] ) * 1.3] )
    
    Fig2.tight_layout() # prevents the plot from clipping ticks
    plt.subplots_adjust( hspace = 0.001 )
    
    # ------------------------------------------------------------------------------ Console Printing ------------------------------------------------------------------------------
    # print summary to console
    print( Imposed.shape[1], "Terms yielding an Mean absolute Error (MAE) of", MeanAbsErrorStr + "% and a maximal deviation of", MaxDeviationStr +
          "% and a Median Absolute Deviation (MAD) of", MedianAbsDerivationStr )
    
    if ( PrintRegressor ): # print the regressors in a readable manner
      print( "\nRecognized regressors:" )
      for i in range( len( SortedERR ) ): print( self.theta[ list( Order ) ][i].item(), RegNames[i] )
      print( "\n" )

    return ( ( Fig, Fig2 ), ( Ax, Ax2 ) ) 