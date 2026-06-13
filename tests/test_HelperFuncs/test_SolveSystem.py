import pytest
import torch
from typing import Tuple
from NARMAX.HelperFuncs import SolveSystem # adjust import

# ---------- helpers ----------
def dense_to_list( U: torch.Tensor ) -> list:
  n = U.shape[ 0 ]
  return [ U[ : j, j ].clone() for j in range( 1, n ) ]


def random_system( n: int, dtype: torch.dtype = torch.float64 ) -> Tuple[ list, list, torch.Tensor ]:
  U = torch.triu( torch.randn( n, n, dtype = dtype ), diagonal = 1 )
  U += torch.eye( n, dtype = dtype )
  x_expected = torch.randn( n, dtype = dtype )
  b = U @ x_expected
  A_list = [ U[ : j, j ].clone() for j in range( 1, n ) ]
  return A_list, b.tolist(), x_expected

# ---------- tests ----------
class TestSolveSystem:
  @pytest.mark.parametrize( "n", [ 1, 2, 5, 10 ] )
  def test_correct_solution( self, n: int ) -> None:
    '''Correct solution for various system sizes.'''
    A_list, W, expected = random_system( n )
    result = SolveSystem( A_list, W )
    # For the original solver we expect exactly the same behaviour:
    # dtype/device of result must match what the original produced

    # n==1 is an exceptino since A is empty and we can't deduce the device and default to cpu as safety
    if ( n == 1 ): expected = torch.tensor( W, dtype = torch.float64, device = torch.device( "cpu" ) )

    assert isinstance( result, torch.Tensor )
    assert result.ndim == 1, f"expected 1D, got { result.ndim }D"
    assert result.shape[ 0 ] == n, f"expected shape ({ n },), got { result.shape }"
    assert result.dtype == expected.dtype, "dtype mismatch"
    assert result.device == expected.device, "device mismatch"
    assert torch.allclose( result, expected, atol = 1e-6 )

  def test_empty_A_list( self ) -> None:
    '''Empty A_list returns W as tensor.'''
    W = [ 42.0 ]
    result = SolveSystem( [], W )
    expected = torch.tensor( [ 42.0 ], dtype = torch.float64, device = torch.device( "cpu" ) )
    assert isinstance( result, torch.Tensor )
    assert result.ndim == 1
    assert result.shape[ 0 ] == 1
    assert result.device == expected.device
    assert result.dtype == expected.dtype
    assert torch.equal( result, expected )

  def test_wrong_W_length_raises( self ) -> None:
    '''Mismatched W length raises ValueError.'''
    A_list = [ torch.randn( 1 ) ] # n=2
    with pytest.raises( ValueError, match = r"W length \(3\) must equal A_List length \+ 1 \(2\)" ):
      SolveSystem( A_list, [ 1.0, 2.0, 3.0 ] )

  def test_wrong_column_length_raises( self ) -> None:
    '''Wrong column length raises ValueError.'''
    # For n=3, A_list[1] (col=2) should have length 2; we give 5
    good_col1 = torch.randn( 1 ) # col=1, length 1 (ok)
    bad_col2 = torch.randn( 5 ) # col=2, length should be 2
    A_list = [ good_col1, bad_col2 ]
    with pytest.raises( ValueError, match = r"A_List\[1\] length \(5\) should be 2" ):
      SolveSystem( A_list, [ 1.0, 2.0, 3.0 ] )

  def test_non_1d_tensor_raises( self ) -> None:
    '''Non-1D tensor in A_list raises ValueError.'''
    A_list = [ torch.randn( 1, 1 ) ]
    with pytest.raises( ValueError, match = r"A_List\[0\] must be 1D, got 2D" ):
      SolveSystem( A_list, [ 1.0, 2.0 ] )

  def test_mixed_devices_raises( self ) -> None:
    '''Mixed device tensors raise ValueError.'''
    if ( torch.cuda.is_available() ):
      col1 = torch.randn( 1, device = "cpu" )
      col2 = torch.randn( 2, device = "cuda:0" )
      with pytest.raises( ValueError, match = "same device" ):
        SolveSystem( [ col1, col2 ], [ 1.0, 2.0, 3.0 ] )

  def test_mixed_dtypes_raises( self ) -> None:
    '''Mixed dtype tensors raise ValueError.'''
    col1 = torch.randn( 1, dtype = torch.float32 )
    col2 = torch.randn( 2, dtype = torch.float64 )
    with pytest.raises( ValueError, match = r"same device and dtype\. Mismatch in column 1" ):
      SolveSystem( [ col1, col2 ], [ 1.0, 2.0, 3.0 ] )

  def test_device_inherited_from_A_List( self ) -> None:
    '''Result device matches A_list device.'''
    if ( torch.cuda.is_available() ):
      device = torch.device( "cuda:0" )
      A_list = [ torch.randn( 1, device = device ), torch.randn( 2, device = device ) ]
      result = SolveSystem( A_list, [ 1.0, 2.0, 3.0 ] )
      assert isinstance( result, torch.Tensor )
      assert result.ndim == 1
      assert result.shape[ 0 ] == 3
      assert result.device == device
      assert result.dtype == A_list[ 0 ].dtype

  def test_double_precision( self ) -> None:
    '''Double precision computation is accurate.'''
    A_list, W, expected = random_system( 10, dtype = torch.float64 )
    result = SolveSystem( A_list, W )
    assert result.dtype == torch.float64
    assert torch.allclose( result, expected, atol = 1e-12 )

  def test_single_precision( self ) -> None:
    '''Single precision computation is accurate.'''
    A_list, W, expected = random_system( 10, dtype = torch.float32 )
    result = SolveSystem( A_list, W )
    assert result.dtype == torch.float32
    assert torch.allclose( result, expected, atol = 1e-5 )

  def test_large_system_residual( self ) -> None:
    '''Large system has small relative residual.'''
    n = 200 # smaller for speed, but still non-trivial
    A_list, W, x_expected = random_system( n, dtype = torch.float64 )
    result = SolveSystem( A_list, W )

    # Reconstruct U from A_list
    U = torch.eye( n, dtype = torch.float64 )
    for j in range( 1, n ): U[ : j, j ] = A_list[ j - 1 ]
    b = torch.tensor( W, dtype = torch.float64 )
    residual = U @ result - b
    rel_res = torch.linalg.norm( residual ) / torch.linalg.norm( b )
    assert rel_res < 1e-6, f"Relative residual too large: { rel_res }"

  def test_reproducibility( self ) -> None:
    '''Same inputs produce same result.'''
    A_list, W, _ = random_system( 5, dtype = torch.float64 )
    out1 = SolveSystem( A_list, W )
    out2 = SolveSystem( A_list, W )
    assert torch.equal( out1, out2 )

  def test_result_satisfies_system( self ) -> None:
    '''Solution x satisfies U @ x == W.'''
    n = 5
    A_list, W, _ = random_system( n, dtype = torch.float64 )
    result = SolveSystem( A_list, W )
    U = torch.eye( n, dtype = torch.float64 )
    for j in range( 1, n ): U[ : j, j ] = A_list[ j - 1 ]
    b = torch.tensor( W, dtype = torch.float64 )
    assert torch.allclose( U @ result, b, atol = 1e-12 )

  def test_integer_W_works( self ) -> None:
    '''Integer values in W list are accepted.'''
    A_list = [ torch.tensor( [ 2.0 ], dtype = torch.float64 ),
               torch.tensor( [ 3.0, 4.0 ], dtype = torch.float64 ) ]
    W = [ 1, 2, 3 ]  # ints, not floats
    result = SolveSystem( A_list, W )
    assert isinstance( result, torch.Tensor )
    assert result.ndim == 1
    assert result.shape[ 0 ] == 3
    assert result.dtype == torch.float64

  def test_empty_W_raises( self ) -> None:
    '''Empty W list with non-empty A_list raises ValueError.'''
    A_list = [ torch.randn( 1 ) ]
    with pytest.raises( ValueError, match = r"W length \(0\) must equal A_List length \+ 1 \(2\)" ):
      SolveSystem( A_list, [] )

  def test_both_empty_raises( self ) -> None:
    '''Empty A_list and empty W raises ValueError.'''
    with pytest.raises( ValueError, match = r"W length \(0\) must equal A_List length \+ 1 \(1\)" ):
      SolveSystem( [], [] )
