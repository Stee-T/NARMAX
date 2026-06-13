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
import math # needed for the binomial coefficient

from typing import Sequence, Union, List
from numpy.typing import NDArray

############################################################################# Set Tensortype And Device ############################################################################
def Set_Tensortype_And_Device() -> str:
  '''Set the default dtype and device for torch tensors. Returns the device string for further use.

  ### Outputs
  - Device: String, either "cpu", "mps", "vulkan", "opencl" or "cuda"
  '''
  if ( tor.cuda.is_available() ):
    Device: str = "cuda" # force new tensors to be on GPU
    tor.set_default_dtype( tor.float64 ) # force new tensors to be 64-bit floats

  elif ( tor.backends.mps.is_available() ):
    Device: str = "mps" # Mac with MXXX (M1, etc.)
    tor.set_default_dtype( tor.float32 ) # force new tensors to be 32-bit floats doesn't support 64-bit

  # elif ( tor.has_opencl() ):                   Device: str = "opencl"  # Devices with OpenCL support
  # elif ( tor.backends.vulkan.is_available() ): Device: str = "vulkan"  # Vulkan devices
  # elif ( tor.backends.mkl.is_available() ):    Device: str = "mkl" # Intel MKL backend

  else:
    print( "\n\nYour python installation didn't detect a hardware-accelerator (CUDA, MPS, Vulkan), so this will run on CPU which is a lot slower\n\n" )
    Device: str = "cpu" # force new tensors to be on CPU
    tor.set_default_dtype( tor.float64 ) # force new tensors to be 64-bit floats irrespective of CPU/GPU/M1/M2

  tor.set_default_device( Device )

  return ( Device )


#################################################################################### Combinations ##################################################################################
def Combinations( N: int, k: int ) -> int:
  '''
  Number of ways to choose k elements from N (binomial coefficient).

  Only non-negative integer arguments are accepted.
  Returns 0 if k > N.
  Raises ValueError for negative inputs or non-integer types.
  '''
  if ( ( not isinstance( N, int ) ) or ( not isinstance( k, int ) ) ): raise TypeError( "N and k must be integers" )
  if ( ( N < 0 ) or ( k < 0 ) ): raise ValueError( "N or k is negative, which is not supported" )
  if ( k > N ): return 0 # R[1/2]
  return math.comb( N, k ) # R[2/2]

####################################################################################### CutY #######################################################################################
def CutY( y: tor.Tensor, Lags: int | Sequence ) -> tor.Tensor:
  '''
  Trim the front of the output signal `y` to the overall maximum lag.

  `Lags` may be:
    - an int,
    - a flat sequence of ints,
    - a sequence containing ints and/or **one** level of nested flat sequences of ints.
  Empty top-level sequences are not allowed (no trimming).
  Nested empty sequences are forbidden (they are ambiguous).

  Raises
  ------
  TypeError
      If `y` is not a 1-D torch tensor, or if any element of `Lags` (or sub-element) is not an integer.
  ValueError
    If `Lags` is empty
    If `y` is not 1-D, any lag is negative, or a nested sequence is empty.
  '''
  # --- Validate y ---
  if ( not isinstance( y, tor.Tensor ) ): raise TypeError( "y must be a torch tensor" )
  if ( y.ndim != 1 ): raise ValueError( "y must be 1-D" )

  # --- Normalise Lags ---
  if ( isinstance( Lags, int ) ):
    if ( Lags < 0 ): raise ValueError( "Lags must be non-negative when given as int" )
    q = Lags
  elif ( ( isinstance( Lags, Sequence ) ) and ( not isinstance( Lags, ( str, bytes ) ) ) ): # Top-level empty sequence → no trimming
    if ( len( Lags ) == 0 ): raise ValueError( "Empty Lags is not allowed" )
    else:
      q = 0
      for lag in Lags:
        if ( isinstance( lag, int ) ):
          if ( lag < 0 ): raise ValueError( f"Lags must be non-negative, got { lag }" )
          q = max( q, lag )
        elif ( ( isinstance( lag, Sequence ) ) and ( not isinstance( lag, ( str, bytes ) ) ) ):
          # One level of nesting (e.g., [1,2,3])
          if ( len( lag ) == 0 ): raise ValueError( "Nested empty sequence in Lags is not allowed" )
          # All elements inside must be integers
          for sublag in lag:
            if ( not isinstance( sublag, int ) ): raise TypeError( f"All elements inside nested sequence must be integers, got { type( sublag ).__name__ }" )
            if ( sublag < 0 ):                  raise ValueError( f"Lags must be non-negative, got { sublag }" )
          # safe to compute max
          q = max( q, max( lag ) )
        else: raise TypeError( f"Each item in Lags must be an int or a flat sequence of ints, got { type( lag ).__name__ }" )
  else: raise TypeError( f"Lags must be an int or a sequence of ints, got { type( Lags ).__name__ }" )

  return y[ q : ]

############################################################################### Squared norm function ##############################################################################
def Norm2( x: tor.Tensor, epsilon: float = 1e-12 ) -> Union[ float, tor.Tensor ]:
  ''' Dimension-aware squared Euclidean norm, more efficient than using torch.norm and squaring because it avoids the square root.
  Values below epsilon are replaced by epsilon since the norm is typically used in denominators.
  '''
  # ---- Input checks
  if ( not isinstance( x, tor.Tensor ) ): raise TypeError( f"Norm2 expects a torch.Tensor, got { type( x ).__name__ }" )
  if ( x.ndim == 0 ): raise ValueError( "Norm2 does not support scalar (0-dimensional) tensors" )
  if ( tor.isnan( x ).any() ): raise ValueError( "Input tensor contains NaN values" )

  # ---- Core logic
  # Single (p,)-array or column ( p,1 ) array, return scalar
  if ( ( x.ndim == 1 ) or ( x.ndim == 2 and x.shape[ 1 ] == 1 ) ): return max( float( tor.sum( x**2 ).item() ), epsilon ) # R[1/2] cast to float in case input tensor isn't, also float is 64 bit
  if ( x.ndim == 2 ):
    # Column-wise squared norm, shape (1, n)
    Out: tor.Tensor = tor.sum( x**2, dim = 0, keepdim = True ) # columnwise for matrices, return ( 1,n ) array being one norm per column
    Out[ Out < epsilon ] = epsilon # replace by a fudge factor to prevent division by quasi-zero
    return Out # R[2/2]

  raise AssertionError( "This function isn't supposed to receive tensors of degree > 2" )

########################################################## Column deleter function (used by recursive FOrLSr matrix update) ########################################################
def DeleteColumn( Tensor: tor.Tensor, index: int ) -> tor.Tensor:
  '''Delete a column from a 2D tensor. Raises IndexError for negative indices.
  This was tested against integer indexing, bool indexing, tor.index_select and somehow this is the fastest.
  '''
  if ( Tensor.dim() < 2 ): raise ValueError( "Tensor must be at least 2D" )
  if ( ( index < 0 ) or ( index >= Tensor.shape[ 1 ] ) ): raise IndexError( f"Column index { index } out of bounds for { Tensor.shape[ 1 ] } columns" )

  if ( index == Tensor.shape[ 1 ] - 1 ): return Tensor[ :, : -1 ] # R[1/2] Last column → return view without copying
  return tor.cat( ( Tensor[ :, : index ], Tensor[ :, index + 1 : ] ), dim = 1 ) # R[2/2] All other columns → concatenate slices (copies data)


###################################################################### SolveSystem (used by recursive FOrLSr) #####################################################################
def SolveSystem( A_List: List[ tor.Tensor ], W: List[ float ] ) -> tor.Tensor:
  '''
  Solves the sparse upper unitriangular system using a dense matrix.
  A_List contains strictly upper-triangular elements column by column.
  '''
  nCols: int = len( A_List ) + 1

  # --- Input validation ---
  if ( len( W ) != nCols ): raise ValueError( f"W length ({ len( W ) }) must equal A_List length + 1 ({ nCols })" )

  # Determine device / dtype from the first tensor (if nCols > 1)
  if ( nCols > 1 ):
    device = A_List[ 0 ].device
    dtype = A_List[ 0 ].dtype
  else: # nCols == 1, thus A == []. We have no information on the device so we default to "cpu" as safety
    device = "cpu"
    dtype = tor.float64

  # Check each column tensor
  for idx, Column in enumerate( A_List, start = 1 ):
    if ( Column.ndim != 1 ):       raise ValueError( f"A_List[{ idx - 1 }] must be 1D, got { Column.ndim }D" )
    if ( Column.shape[ 0 ] != idx ): raise ValueError( f"A_List[{ idx - 1 }] length ({ Column.shape[ 0 ] }) should be { idx }" )
    if ( ( nCols > 1 ) and ( Column.device != device or Column.dtype != dtype ) ):
      raise ValueError( f"All tensors in A_List must have the same device and dtype. Mismatch in column { idx - 1 }." )

  # Build dense matrix & solve: it's ok to build it here because we know the exact size and only less than half the entrie are 0s (upper triangular).
  A_Mat = tor.eye( nCols, dtype = dtype, device = device )
  for col in range( 1, nCols ): A_Mat[ : col, col ] = tor.ravel( A_List[ col - 1 ] )
  b = tor.tensor( W, dtype = dtype, device = device ).view( -1, 1 )

  return tor.ravel( tor.linalg.solve_triangular( A_Mat, b, upper = True, unitriangular = True ) ) # theta


################################################################################# Remove Duplicates ################################################################################
def RemoveDuplicates( RegMat: tor.Tensor, RegNames: NDArray[ np.str_ ] ) -> tuple[ tor.Tensor, NDArray[ np.str_ ], NDArray[ np.int64 ] ]:
  '''
  OOM-safe exact duplicate column removal. Preserves input device.
  Auxiliary memory: O(n_cols). Zero full-matrix copies.
  Deterministic: fingerprinting is a fast filter; exact verification guarantees correctness.
  Requires PyTorch >= 1.12.
  '''

  if ( RegMat.dim() != 2 ): raise ValueError( "RegMat must be 2D (n_rows, n_cols)" )
  if ( tor.isnan( RegMat ).any() ): raise ValueError( "RegMat contains NaNs" )
  if ( tor.isinf( RegMat ).any() ): raise ValueError( "RegMat contains Infs" )

  n_rows: int = RegMat.shape[ 0 ]
  n_cols: int = RegMat.shape[ 1 ]

  if ( n_cols == 0 ): return RegMat, RegNames, np.array( [], dtype = np.int64 ) # R[1/3]
  if ( n_rows == 0 ): return RegMat[ :, : 1 ], RegNames[ : 1 ], np.array( [ 0 ], dtype = np.int64 ) # R[2/3]

  # Deterministic projections for O(n_cols) memory fingerprinting.
  # Precision matches RegMat.dtype to strictly avoid OOM from full-matrix upcasting.
  # Lower precision increases collision frequency, but exact verification ensures correctness.
  proj_dtype: tor.dtype = tor.float64 if RegMat.dtype == tor.float64 else tor.float32
  v1: tor.Tensor = tor.arange( 1, n_rows + 1, device = RegMat.device, dtype = proj_dtype )
  v2: tor.Tensor = v1 * v1

  # Matrix-vector product: O(n_rows * n_cols) compute, O(n_cols) memory.
  fp1: tor.Tensor = tor.mv( RegMat.T, v1.to( RegMat.dtype ) )
  fp2: tor.Tensor = tor.mv( RegMat.T, v2.to( RegMat.dtype ) )
  sigs: tor.Tensor = tor.stack( [ fp1, fp2 ], dim = 1 )

  # Group by signature
  _, inv = tor.unique( sigs, dim = 0, return_inverse = True )
  inv = inv.view( -1 )
  num_groups: int = int( inv.max() ) + 1

  # Identify groups requiring exact verification
  counts: tor.Tensor = tor.bincount( inv, minlength = num_groups )
  collision_groups: tor.Tensor = tor.nonzero( counts > 1 ).view( -1 )

  # Track duplicates exactly. Initialize all as unique.
  is_duplicate: tor.Tensor = tor.zeros( n_cols, dtype = tor.bool, device = RegMat.device )

  if ( collision_groups.numel() > 0 ):
    for gid in collision_groups.tolist():
      members: tor.Tensor = tor.nonzero( inv == gid ).view( -1 )
      # Exact pairwise verification within collision group.
      # Retains false positives (distinct columns with identical signatures).
      group_kept: list[ int ] = [ members[ 0 ].item() ]
      for m_idx in range( 1, members.numel() ):
        m: int = members[ m_idx ].item()
        is_exact_dup: bool = False
        for k in group_kept:
          if ( tor.equal( RegMat[ :, m ], RegMat[ :, k ] ) ):
            is_exact_dup = True
            break
        if ( is_exact_dup ): is_duplicate[ m ] = True
        else: group_kept.append( m )

  keep_idx: tor.Tensor = tor.nonzero( ~is_duplicate ).view( -1 )

  n_redundant: int = n_cols - len( keep_idx )
  if ( n_redundant > 0 ):
    suffix: str = "s were" if n_redundant > 1 else " was"
    print( f"\n\nWARNING: { n_redundant } redundant regressor{ suffix } eliminated from the dictionary. Use the returned one for further operations.\n\n" )

  idx_np: NDArray[ np.int64 ] = keep_idx.cpu().numpy()
  return RegMat.index_select( 1, keep_idx ), RegNames[ idx_np ], idx_np # R[3/3]

################################################################################# All Combinations #################################################################################
def AllCombinations( ImposedRegs: NDArray[ np.int64 ], InputSeq: NDArray[ np.int64 ], INT_TYPE: np.dtype ) -> NDArray[ np.int64 ]:
  '''
  Helper function creating all combinations of imposed regressors to construct the current node's children. Assumes that InputSeq has been flattened and
  contains **no duplicates**. ImposedRegs may contain duplicates (they will be repeated in every output row).

  Parameters
  ----------
  ImposedRegs : ndarray of int64 Imposed regressors that will appear in all output rows.
  InputSeq : ndarray of int64
      Flattened sequence of candidate regressors. Must have no duplicates.
  INT_TYPE : dtype
      Desired integer dtype for the output array. Must be able to safely hold
      all values from the inputs (e.g., int64 if inputs are int64).

  Returns
  -------
  out : ndarray of INT_TYPE
      Array of shape (n_new_regressors, len(ImposedRegs)+1). Each row contains the imposed regressors followed by one new regressor from InputSeq
      (with elements already present in ImposedRegs removed).
  '''
  # --- input validation ---
  # 1. Ensure INT_TYPE can safely represent the input values.
  if ( not np.can_cast( ImposedRegs.dtype, INT_TYPE, casting = "safe" ) ):
    raise ValueError( f"INT_TYPE ({ INT_TYPE }) cannot safely hold values from ImposedRegs (dtype { ImposedRegs.dtype })." )
  if ( not np.can_cast( InputSeq.dtype, INT_TYPE, casting = "safe" ) ):
    raise ValueError( f"INT_TYPE ({ INT_TYPE }) cannot safely hold values from InputSeq (dtype { InputSeq.dtype })." )

  # 2. InputSeq must contain no duplicates (required by setdiff1d with assume_unique=True).
  if ( ( InputSeq.size > 0 ) and ( np.unique( InputSeq ).size != InputSeq.size ) ): raise ValueError( "InputSeq must not contain duplicate values." )

  # Remove elements from InputSeq that are already in ImposedRegs. # With assume_unique=True, the original order of InputSeq is preserved.
  InputSeq = np.setdiff1d( InputSeq, ImposedRegs, assume_unique = True )

  Out: NDArray[ np.int64 ] = np.empty( ( len( InputSeq ), len( ImposedRegs ) + 1 ), dtype = INT_TYPE ) # dimensions are ( nNewRegressors=nCombinations, Combination lengh )
  Out[ :, : -1 ] = ImposedRegs # all rows start with the imposed terms
  Out[ :, -1 ] = InputSeq # the last columns are the new regressors
  return ( Out )
