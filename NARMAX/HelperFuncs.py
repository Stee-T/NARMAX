"""
This place is a message... and part of a system of messages... pay attention to it! Sending this message was important to us. We considered ourselves to be a powerful culture.
This place is not a place of honor...no highly esteemed deed is commemorated here... nothing valued is here.

What is here is dangerous and repulsive to us. This message is a warning about danger.

The danger is in a particular location... it increases toward a center... the center of danger is here... of a particular size and shape, and below us.

The danger is still present, in your time, as it was in ours. The danger is to the body, and it can kill.
The form of the danger is an emanation of energy.
The danger is unleashed only if you substantially disturb this place physically. This place is best shunned and left uninhabited.
"""

import torch as tor # general operations
import numpy as np # needed for the duplicate elimination

# ***************************************************************************** Set Tensortype And Device  ******************************************************************************
def Set_Tensortype_And_Device():
  '''Set the default dtype and device for torch tensors. Returns the device string for further use.
  
  ### Outputs
  - Device: String, either "cpu", "mps", "vulkan", "opencl" or "cuda"
  '''
  if ( tor.cuda.is_available() ):            Device = "cuda" # force new tensors to be on GPU
  elif ( tor.backends.mps.is_available() ):    Device = "mps" # M1/M2 Macs
  # elif ( tor.has_opencl() ):                     Device = "opencl"  # Devices with OpenCL support
  # elif ( tor.backends.vulkan.is_available() ): Device = "vulkan"  # Vulkan devices
  # elif ( tor.backends.mkl.is_available() ):    Device = "mkl" # Intel MKL backend
  else:
    print( "\n\nYour python installation didn't detect a hardware-accelerator (CUDA, MPS, Vulkan), so this will run on CPU which is a lot slower\n\n" )
    Device = "cpu" # force new tensors to be on CPU
  
  tor.set_default_device( Device )
  tor.set_default_dtype( tor.float64 ) # force new tensors to be 64-bit floats irrespective of CPU/GPU/M1/M2

  return ( Device )
  

# *********************************************************************************** Combinations ****************************************************************************************
def Combinations( N, k ):
  '''From N chose k for positive values only'''
  if ( k > N ): return ( 0 ) # property of the operator
  if ( k < 0 or N < 0 ): raise ValueError( "N or k is negative, which is not supported" )

  def factorial( n ):
    fact = 1 # base case, guaranteed output even if looop not entered
    for i in range( 2,int( n )+1 ): fact *= i # multiply upwards, +1 due to python indexing
    return ( fact )

  return ( int( factorial( N ) / ( factorial( k ) * factorial( N - k ) ) ) ) # N!/( k!*( N-k )! )


# *********************************************************************************** CutY ****************************************************************************************
def CutY( y, Lags ):
  '''Function to trim the y vector data to the maximum lag in Lags. The front is cut as would do the Lagger CTor
  
  ### Inputs
  - y: (1D torch tensor) containing the system output
  - Lags: (int or iterable of ints) containing the maximum lags of each system input
  
  ### Outputs
  - y: (1D torch tensor) cut to Maxlags
  '''
  
  if ( isinstance( Lags, int ) ): Lags = ( Lags, )
  if ( y.ndim != 1 ): raise ValueError( "y must be 1D" )
  
  q = 0 # Maximum lag to trim all regressors to
  for lag in Lags:
    if ( isinstance( lag, int ) ): q = max( q, lag )
    else:                          q = max( q, max( lag ) ) # iterable can contain whatever

  return ( y[ q : ] )



# *********************************************************************************** FindMinInt ****************************************************************************************
def FindMinInt( nCols ):
  '''Function determining the numpy integer dtype necessary to hold the current nCols to reduce memory usage'''
  if   ( nCols <= np.iinfo( np.uint16 ).max ): return ( np.uint16 )
  elif ( nCols <= np.iinfo( np.uint32 ).max ): return ( np.uint32 ) # certainly sufficient in most cases
  else: return ( np.uint64 )

# ****************************************************************************** Squared norm function ********************************************************************************
def Norm2( x, epsilon = 1e-12 ):
  ''' Dimension aware Squared euclidean norm, more efficient than using tor.norm and squaring due to no sqrt. Overwrites with fudge factor since norm are used as divisions'''
  if ( ( x.ndim == 1 ) or ( x.ndim == 2 and x.shape[1] == 1 ) ): return ( max( [tor.sum( x**2 ).item(), epsilon] ) ) # Single (p,)-array or column ( p,1 ) array, return scalar
  if ( x.ndim == 2 ):
    Out = tor.sum( x**2, axis = 0, keepdims = True ) # columnwise for matrices, return ( 1,n ) array being one norm per column
    Out[Out < epsilon] = epsilon # replace near zero values by a fudge factor to prevent division by zero
    return ( Out )
  else: raise AssertionError( "This function isn't supposed to get any tensors of degree > 2" )


# ************************************************************************************************ Column deleter function (used by recusrive FOrLSr matrix update) ********************************************************************************
def DeleteColumn( Tensor, index ):
  '''Deletes a column from a torch tensor, since that doesn't seem to exist in pytorch'''
  if ( index == Tensor.shape[1] - 1 ): return ( Tensor[:, :-1] ) # exclude last column
  else: return ( tor.hstack( (Tensor[:, :index], Tensor[:, index+1:]) ) ) # concat around missing column


# ************************************************************************************************ SolveSuystem (used by recusrive FOrLSr) ********************************************************************************
def SolveSystem( A_List, W ):
  '''Gets A in the 1-tensor per column format and solves the sparse upper unitriangular system.
  
  ### Output:
  - `theta`: ( torch tensor ) containing the system solution coefficients'''

  A_Mat = tor.eye( len( A_List ) + 1 ) # square matrix, has one more row+col than A_List, since the first entry is a 1 ( unidiagonal )
  for col in range( 1, A_Mat.shape[1] ): A_Mat[:col, col] = tor.ravel( A_List[col-1] ) # Copy into upper triangular matrix
  theta = tor.linalg.solve_triangular( A_Mat, tor.tensor( W ).view(-1,1), upper = True, unitriangular = True ) # Get regressor coefficients

  return ( tor.ravel( theta ) )


# ************************************************************************************************ RemoveDuoplicates ********************************************************************************
def RemoveDuplicates( RegMat, RegNames ):
  '''This function is required since torch unique sorts the data (might change though)
  ### Inputs:
  - `RegMat`: ( ( n, p )-sized float torch tensor ) containing the regressors columnwise
  - `RegNames`: ( ( n, )-sized numpy array ) containing the regressor names
  
  ### Outputs:
  - `RegMat`: Filtered Regressormatrix of <= dimension than RegMat
  - `RegNames`: Filtered Regressornames of <= dimension than RegNames
  - `DcFilterIdx`: Indexset of the remaining regressors
  '''
  # Clean the dictionary of equivalent entries to speed up the search and avoid ( +inf - inf ) problems during fitting
  nCols = RegMat.shape[1]
  TempData = ( RegMat.cpu().numpy() ).T # copy to CPU memory (if necessary) and Cast to numpy
  b = np.ascontiguousarray( TempData ).view( np.dtype( ( np.void, TempData.dtype.itemsize * TempData.shape[1] ) ) ) # black magic recast
  _, DcFilterIdx, indices = np.unique( b, return_index = True, return_inverse = True )
  DcFilterIdx = np.sort( DcFilterIdx ) # keep the original Data order

  if ( len( DcFilterIdx ) < nCols ): # Dictionary (Ds or Dc) was filtered, warn user.
    nRedundant = nCols - len( DcFilterIdx )
    print( f"\n\nWARNING: { nRedundant } redundant regressor{ "s were" if nRedundant > 1 else " is" } eliminated from the dictionary. Use the returned one for further operations.\n\n" )

  return ( ( RegMat.T[DcFilterIdx] ).T, RegNames[DcFilterIdx], DcFilterIdx ) # Apply filter on original GPU tensor


# ******************************************************************************************** All Combinations ********************************************************************************************
def AllCombinations( ImposedRegs, InputSeq, INT_TYPE ): 
  '''Helper function creating all combiantions of imposed Regressors to construct the current node's children. Assumes that InputSeq has been flattened'''
  InputSeq = np.setdiff1d( InputSeq, ImposedRegs, True ) # make sure InputSeq doesn't contain the imposed terms (only needed for OOIT-predicted matches)
  # TODO this works but verify that OOIT predicted matches have the property that the imposed terms aren't necessary in the beginning

  Out = np.empty( ( len( InputSeq ), len( ImposedRegs ) + 1 ), dtype = INT_TYPE ) # dimensions are ( nNewRegressors=nCombinations, Combination lengh )
  Out[:, :-1] = ImposedRegs # all rows start with the imposed terms
  Out[:, -1] = InputSeq # the last columns are the new regressors
  return ( Out )