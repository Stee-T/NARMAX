import math
import pytest
import torch as tor
import numpy as np
from typing import Union

from NARMAX.HelperFuncs import Norm2 # adjust import

# -------------------------------------------------------------------
# Helpers to compare tensor/float output with expected value
# -------------------------------------------------------------------
def _val( actual: Union[ tor.Tensor, float, int ] ) -> float:
  '''Extract Python float from either a float/int or a torch.Tensor.'''
  if ( isinstance( actual, tor.Tensor ) ): return actual.item() # R[1/2]
  return float( actual ) # R[2/2]


def assert_close( actual: tor.Tensor, expected: tor.Tensor ) -> None:
  '''Assert that actual and expected tensors are close.'''
  assert isinstance( actual, tor.Tensor )
  assert isinstance( expected, tor.Tensor )
  assert tor.allclose( actual, expected )


def assert_scalar_close( actual: Union[ tor.Tensor, float, int ], expected: float, eps: float = 1e-12 ) -> None:
  '''Assert that a scalar (float/int/tensor) is close to a float.'''
  assert abs( _val( actual ) - expected ) < eps


# ===================================================================
# 1. Input validation (new checks)
# ===================================================================
class TestInputValidation:
  def test_rejects_non_tensor( self ) -> None:
    '''Non-tensor input raises TypeError with message.'''
    with pytest.raises( TypeError, match = "Norm2 expects a torch.Tensor" ):
      Norm2( [ 1.0, 2.0 ] )

  def test_rejects_scalar_tensor( self ) -> None:
    '''Scalar tensor raises ValueError with message.'''
    x = tor.tensor( 5.0 )
    with pytest.raises( ValueError, match = "Norm2 does not support scalar" ):
      Norm2( x )

  def test_rejects_3d_tensor( self ) -> None:
    '''3D tensor raises AssertionError.'''
    x = tor.ones( 2, 3, 4 )
    with pytest.raises( AssertionError, match = "degree > 2" ):
      Norm2( x )

  def test_rejects_4d_tensor( self ) -> None:
    '''4D tensor raises AssertionError.'''
    x = tor.ones( 2, 3, 4, 5 )
    with pytest.raises( AssertionError, match = "degree > 2" ):
      Norm2( x )

  def test_rejects_nan_1d( self ) -> None:
    '''1D tensor with NaN raises ValueError.'''
    x = tor.tensor( [ 1.0, float( 'nan' ) ] )
    with pytest.raises( ValueError, match = "NaN" ):
      Norm2( x )

  def test_rejects_nan_2d( self ) -> None:
    '''2D tensor with NaN raises ValueError.'''
    x = tor.tensor( [ [ float( 'nan' ), 1.0 ], [ 2.0, 3.0 ] ] )
    with pytest.raises( ValueError, match = "NaN" ):
      Norm2( x )

  def test_rejects_nan_in_column_vector( self ) -> None:
    '''Column vector with NaN raises ValueError.'''
    x = tor.tensor( [ [ float( 'nan' ) ], [ 1.0 ] ] )
    with pytest.raises( ValueError, match = "NaN" ):
      Norm2( x )


# ===================================================================
# 2. Basic correctness (float tensors)
# ===================================================================
class TestBasic:
  def test_1d_single_element( self ) -> None:
    '''1D single element computes squared norm.'''
    x = tor.tensor( [ 3.0 ] )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 9.0 )

  def test_1d_multiple_elements( self ) -> None:
    '''1D multiple elements computes sum of squares.'''
    x = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 14.0 )

  def test_1d_all_zeros_uses_epsilon( self ) -> None:
    '''All-zero 1D input returns epsilon.'''
    x = tor.zeros( 5 )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-12 )

  def test_1d_custom_epsilon( self ) -> None:
    '''Custom epsilon parameter works for 1D input.'''
    x = tor.zeros( 5 )
    out = Norm2( x, epsilon = 1e-6 )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-6 )

  def test_2d_single_column( self ) -> None:
    '''2D single column computes sum of squares.'''
    x = tor.tensor( [ [ 1.0 ], [ 2.0 ], [ 3.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 14.0 )

  def test_2d_single_column_zeros( self ) -> None:
    '''All-zero single column returns epsilon.'''
    x = tor.zeros( 3, 1 )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-12 )

  def test_2d_matrix_multicolumn( self ) -> None:
    '''Multi-column 2D matrix computes column norms.'''
    x = tor.tensor( [ [ 1.0, 2.0 ],
                          [ 3.0, 4.0 ],
                          [ 5.0, 6.0 ] ] )
    expected = tor.tensor( [ [ 35.0, 56.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, tor.Tensor )
    assert out.shape == ( 1, 2 )
    assert_close( out, expected )

  def test_2d_matrix_some_cols_below_epsilon( self ) -> None:
    '''Columns below epsilon are clamped to epsilon.'''
    x = tor.tensor( [ [ 0.0, 1e-13 ],
                          [ 0.0, 0.0 ] ] )
    out = Norm2( x )
    assert_close( out, tor.tensor( [ [ 1e-12, 1e-12 ] ] ) )

  def test_1d_negative_values( self ) -> None:
    '''1D negative values produce same result as positive.'''
    x = tor.tensor( [ -1.0, -2.0, -3.0 ] )
    out = Norm2( x )
    assert isinstance( out, float )
    assert_scalar_close( out, 14.0 )

  def test_2d_column_negative_values( self ) -> None:
    '''2D column with negative values computes sum of squares.'''
    x = tor.tensor( [ [ -1.0 ], [ -2.0 ], [ -3.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, float )
    assert_scalar_close( out, 14.0 )

  def test_2d_matrix_mixed_signs( self ) -> None:
    '''2D matrix with mixed signs computes per-column squared norms.'''
    x = tor.tensor( [ [ -1.0, 2.0 ],
                      [ 3.0, -4.0 ] ] )
    expected = tor.tensor( [ [ 10.0, 20.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, tor.Tensor )
    assert out.shape == ( 1, 2 )
    assert_close( out, expected )

  def test_1d_with_inf( self ) -> None:
    '''1D with inf values returns inf.'''
    x = tor.tensor( [ 1.0, float( 'inf' ) ] )
    out = Norm2( x )
    assert isinstance( out, float )
    assert math.isinf( out )

  def test_2d_column_with_inf( self ) -> None:
    '''2D column with inf returns inf.'''
    x = tor.tensor( [ [ 1.0 ], [ float( 'inf' ) ] ] )
    out = Norm2( x )
    assert isinstance( out, float )
    assert math.isinf( out )

  def test_2d_matrix_with_inf( self ) -> None:
    '''2D matrix with inf returns inf in that column.'''
    x = tor.tensor( [ [ 1.0, 2.0 ],
                      [ float( 'inf' ), 3.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, tor.Tensor )
    assert out.shape == ( 1, 2 )
    assert math.isinf( out[ 0, 0 ].item() )
    assert_scalar_close( out[ 0, 1 ].item(), 13.0 )

  def test_2d_matrix_one_col_below_one_above( self ) -> None:
    '''Mixed columns: some clamped, some computed.'''
    x = tor.tensor( [ [ 1e-13, 3.0 ],
                          [ 0.0, 4.0 ] ] )
    out = Norm2( x )
    assert_close( out, tor.tensor( [ [ 1e-12, 25.0 ] ] ) )


# ===================================================================
# 3. Shape edge cases
# ===================================================================
class TestShapes:
  def test_empty_1d( self ) -> None:
    '''Empty 1D tensor returns epsilon.'''
    x = tor.empty( 0 )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-12 )

  def test_empty_2d_column_vector( self ) -> None:
    '''Empty column vector returns epsilon.'''
    x = tor.empty( 0, 1 )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-12 )

  def test_empty_2d_matrix_no_columns( self ) -> None:
    '''Empty matrix with zero columns returns empty tensor.'''
    x = tor.empty( 3, 0 )
    out = Norm2( x )
    assert isinstance( out, tor.Tensor )
    assert out.shape == ( 1, 0 )

  def test_1x1_matrix( self ) -> None:
    '''1x1 matrix computes squared value.'''
    x = tor.tensor( [ [ 5.0 ] ] )
    out = Norm2( x )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 25.0 )

  def test_1xn_matrix( self ) -> None:
    '''1xN matrix computes per-column squares.'''
    x = tor.tensor( [ [ 1.0, 2.0, 3.0 ] ] )
    out = Norm2( x )
    assert_close( out, tor.tensor( [ [ 1.0, 4.0, 9.0 ] ] ) )


# ===================================================================
# 4. Return type consistency all returns are tensors
# ===================================================================
class TestReturnTypes:
  def test_1d_returns_scalar( self ) -> None:
    '''1D input returns scalar.'''
    out = Norm2( tor.tensor( [ 1.0, 2.0 ] ) )
    assert isinstance( out, ( float, int ) )

  def test_2d_column_returns_scalar( self ) -> None:
    '''Single-column 2D input returns scalar.'''
    out = Norm2( tor.randn( 5, 1 ) )
    assert isinstance( out, ( float, int ) )

  def test_2d_matrix_returns_tensor( self ) -> None:
    '''Multi-column 2D input returns tensor.'''
    out = Norm2( tor.randn( 4, 3 ) )
    assert isinstance( out, tor.Tensor )
    assert out.ndim == 2
    assert out.shape[ 0 ] == 1

  def test_2d_matrix_preserves_dtype( self ) -> None:
    '''Output tensor dtype matches input dtype for float32.'''
    x = tor.randn( 4, 3, dtype = tor.float32 )
    out = Norm2( x )
    assert out.dtype == tor.float32

  def test_2d_matrix_preserves_device( self ) -> None:
    '''Output tensor is on the same device as input.'''
    x = tor.randn( 4, 3 )
    out = Norm2( x )
    assert out.device == x.device


# ===================================================================
# 5. Epsilon behaviour
# ===================================================================
class TestEpsilon:
  def test_epsilon_default_value( self ) -> None:
    '''Default epsilon is 1e-12.'''
    out = Norm2( tor.zeros( 10 ) )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-12 )

  def test_epsilon_custom_value_1d( self ) -> None:
    '''Custom epsilon works for 1D input.'''
    out = Norm2( tor.zeros( 10 ), epsilon = 1e-8 )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert_scalar_close( out, 1e-8 )

  def test_epsilon_custom_value_2d( self ) -> None:
    '''Custom epsilon works for 2D input.'''
    x = tor.zeros( 5, 4 )
    out = Norm2( x, epsilon = 1e-3 )
    assert_close( out, tor.full( ( 1, 4 ), 1e-3 ) )

  def test_exact_epsilon_boundary( self ) -> None:
    '''Input at epsilon boundary is not clamped.'''
    x = tor.tensor( [ ( 1e-12 )**0.5 ] )
    out = Norm2( x, epsilon = 1e-12 )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert abs( _val( out ) - 1e-12 ) < 1e-15

  def test_sum_just_above_epsilon( self ) -> None:
    '''Sum just above epsilon uses actual value.'''
    x = tor.tensor( [ ( 1.01e-12 )**0.5 ] )
    out = Norm2( x, epsilon = 1e-12 )
    assert isinstance( out, ( float, int, tor.Tensor ) )
    assert _val( out ) > 1e-12

  def test_epsilon_zero_1d_all_zero( self ) -> None:
    '''Epsilon=0 with all-zero 1D returns 0.0.'''
    x = tor.zeros( 5 )
    out = Norm2( x, epsilon = 0.0 )
    assert out == 0.0

  def test_epsilon_zero_2d_column_all_zero( self ) -> None:
    '''Epsilon=0 with all-zero column returns 0.0.'''
    x = tor.zeros( 5, 1 )
    out = Norm2( x, epsilon = 0.0 )
    assert out == 0.0

  def test_epsilon_zero_2d_matrix_all_zero( self ) -> None:
    '''Epsilon=0 with all-zero matrix returns zeros.'''
    x = tor.zeros( 3, 4 )
    out = Norm2( x, epsilon = 0.0 )
    assert_close( out, tor.zeros( 1, 4 ) )

  def test_epsilon_negative_1d_is_noop( self ) -> None:
    '''Negative epsilon has no effect (sum of squares >= 0).'''
    x = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    out = Norm2( x, epsilon = -1.0 )
    assert isinstance( out, float )
    assert_scalar_close( out, 14.0 )

  def test_epsilon_negative_2d_matrix_is_noop( self ) -> None:
    '''Negative epsilon does not clamp non-negative sums.'''
    x = tor.tensor( [ [ 1.0, 0.0 ],
                      [ 2.0, 0.0 ] ] )
    out = Norm2( x, epsilon = -1.0 )
    assert_close( out, tor.tensor( [ [ 5.0, 0.0 ] ] ) )

  def test_epsilon_large_value_2d( self ) -> None:
    '''Large epsilon clamps all columns.'''
    x = tor.tensor( [ [ 1.0, 2.0 ] ] )
    out = Norm2( x, epsilon = 100.0 )
    assert_close( out, tor.tensor( [ [ 100.0, 100.0 ] ] ) )


# ===================================================================
# 6. Behaviour with integer inputs (still allowed but can crash)
# ===================================================================
class TestIntegerInputs:
  def test_integer_1d_works( self ) -> None:
    '''Integer 1D input computes sum of squares.'''
    x = tor.tensor( [ 1, 2, 3 ] )
    out = Norm2( x )
    assert isinstance( out, ( float, int ) )
    assert out == 14.0

  def test_integer_2d_no_error_but_epsilon_lost( self ) -> None:
    '''
    Integer tensor does NOT raise an error because epsilon is a Python
    float, which gets cast to int (0) during in‑place assignment.
    This effectively disables the epsilon safeguard – the zero column
    stays zero instead of becoming epsilon.
    '''
    x = tor.tensor( [ [ 0, 2 ], [ 0, 3 ] ], dtype = tor.int64 )
    out = Norm2( x )
    # Expected: tensor of int64, with first column = 0 (not 1e-12)
    expected = tor.tensor( [ [ 0, 13 ] ], dtype = tor.int64 )
    assert isinstance( out, tor.Tensor )
    assert out.dtype == tor.int64
    assert tor.equal( out, expected )
