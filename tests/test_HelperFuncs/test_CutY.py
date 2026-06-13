import torch as tor
import pytest
import numpy as np
from typing import Union

from NARMAX.HelperFuncs import CutY


class TestCut:

  # ------------------------------------------------------------------
  # Flat sequence cases (int, list, tuple, range) – no empty lags here
  # ------------------------------------------------------------------
  @pytest.mark.parametrize(
        "y, lags, expected",
        [
            ( tor.tensor( [ 1, 2, 3, 4, 5 ] ), 2, tor.tensor( [ 3, 4, 5 ] ) ),
            ( tor.tensor( [ 1, 2, 3, 4, 5 ] ), ( 1, 3 ), tor.tensor( [ 4, 5 ] ) ),
            ( tor.tensor( [ 1, 2, 3, 4, 5 ] ), [ 0, 0 ], tor.tensor( [ 1, 2, 3, 4, 5 ] ) ),
            ( tor.tensor( [ 1, 2, 3, 4, 5 ] ), ( 0, ), tor.tensor( [ 1, 2, 3, 4, 5 ] ) ),
            ( tor.arange( 10 ), range( 2, 5 ), tor.tensor( [ 4, 5, 6, 7, 8, 9 ] ) ),
            ( tor.arange( 5 ), ( 3, ), tor.tensor( [ 3, 4 ] ) ),
            ( tor.tensor( [ 7.0 ] ), 0, tor.tensor( [ 7.0 ] ) ),
        ]
    )
  def test_flat_valid_inputs( self, y: tor.Tensor, lags: Union[ int, tuple, list, range ], expected: tor.Tensor ) -> None:
    '''Flat (non-nested) lags produce correct slice.'''
    result = CutY( y, lags )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, expected )

  # ------------------------------------------------------------------
  # Nested sequence cases (the original MIMO usage)
  # ------------------------------------------------------------------
  def test_nested_list_of_lags( self ) -> None:
    '''Nested list of lags works correctly.'''
    y = tor.arange( 20 )
    lags = ( 3, 3, 3, [ 1, 2, 3 ], [ 1, 2, 3 ] )
    result = CutY( y, lags )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 3 : ] )

  def test_nested_tuple_of_lags( self ) -> None:
    '''Nested tuple of lags works correctly.'''
    y = tor.arange( 20 )
    lags = ( 2, ( 1, 4 ), 5 )
    result = CutY( y, lags )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 5 : ] )

  def test_deeply_nested_raises( self ) -> None:
    '''Deeply nested sequences raise TypeError.'''
    y = tor.tensor( [ 1, 2, 3 ] )
    with pytest.raises( TypeError, match = "All elements inside nested sequence must be integers" ):
      CutY( y, [ 1, [ 2, [ 3 ] ] ] )

  def test_nested_empty_sequence_raises( self ) -> None:
    '''Empty nested tuple raises ValueError.'''
    y = tor.tensor( [ 1, 2, 3 ] )
    with pytest.raises( ValueError, match = "Nested empty sequence" ):
      CutY( y, [ 1, () ] )

  def test_nested_empty_list_raises( self ) -> None:
    '''Empty nested list raises ValueError.'''
    y = tor.tensor( [ 1, 2, 3 ] )
    with pytest.raises( ValueError, match = "Nested empty sequence" ):
      CutY( y, [ 1, [] ] )

  # ------------------------------------------------------------------
  # Edge cases on y
  # ------------------------------------------------------------------
  def test_lag_equals_length( self ) -> None:
    '''Lag equal to tensor length yields empty result.'''
    y = tor.tensor( [ 10, 20, 30 ] )
    result = CutY( y, 3 )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert len( result ) == 0
    assert tor.equal( result, tor.tensor( [], dtype = tor.long ) )

  def test_lag_greater_than_length( self ) -> None:
    '''Lag greater than tensor length yields empty result.'''
    y = tor.tensor( [ 10, 20 ] )
    result = CutY( y, 5 )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert len( result ) == 0
    assert tor.equal( result, tor.tensor( [], dtype = tor.long ) )

  def test_empty_y_tensor( self ) -> None:
    '''Empty y tensor works correctly.'''
    y = tor.tensor( [], dtype = tor.float32 )
    result1 = CutY( y, 2 )
    assert isinstance( result1, tor.Tensor )
    assert result1.dtype == tor.float32
    assert result1.ndim == 1
    assert len( result1 ) == 0
    assert tor.equal( result1, tor.tensor( [], dtype = tor.float32 ) )
    result2 = CutY( y, 0 )
    assert isinstance( result2, tor.Tensor )
    assert result2.dtype == tor.float32
    assert result2.ndim == 1
    assert len( result2 ) == 0
    assert tor.equal( result2, tor.tensor( [], dtype = tor.float32 ) )

  # ------------------------------------------------------------------
  # Data type & immutability
  # ------------------------------------------------------------------
  def test_preserves_dtype( self ) -> None:
    '''Output preserves input dtype.'''
    y = tor.tensor( [ 1.5, 2.5, 3.5 ], dtype = tor.float64 )
    result = CutY( y, 1 )
    assert result.dtype == tor.float64
    assert tor.equal( result, tor.tensor( [ 2.5, 3.5 ], dtype = tor.float64 ) )

  def test_original_tensor_unchanged( self ) -> None:
    '''Original tensor is not modified.'''
    y = tor.tensor( [ 100, 200, 300 ] )
    _ = CutY( y, 1 )
    assert tor.equal( y, tor.tensor( [ 100, 200, 300 ] ) )

  # ------------------------------------------------------------------
  # Invalid y
  # ------------------------------------------------------------------
  def test_y_not_tensor_raises( self ) -> None:
    '''Non-tensor y raises TypeError.'''
    with pytest.raises( TypeError, match = "y must be a torch tensor" ):
      CutY( [ 1, 2, 3 ], 1 )

  def test_y_2d_raises( self ) -> None:
    '''2D y raises ValueError.'''
    with pytest.raises( ValueError, match = "y must be 1-D" ):
      CutY( tor.tensor( [ [ 1, 2 ], [ 3, 4 ] ] ), 1 )

  def test_y_0d_raises( self ) -> None:
    '''0D y raises ValueError.'''
    with pytest.raises( ValueError, match = "y must be 1-D" ):
      CutY( tor.tensor( 42 ), 1 )

  # ------------------------------------------------------------------
  # Invalid Lags – wrong outer type
  # ------------------------------------------------------------------
  def test_lags_float_raises( self ) -> None:
    '''Float lags raises TypeError.'''
    with pytest.raises( TypeError, match = "Lags must be an int or a sequence" ):
      CutY( tor.tensor( [ 1, 2 ] ), 3.14 )

  def test_lags_string_raises( self ) -> None:
    '''String lags raises TypeError.'''
    with pytest.raises( TypeError, match = "Lags must be an int or a sequence" ):
      CutY( tor.tensor( [ 1, 2 ] ), "abc" )

  def test_lags_numpy_array_raises( self ) -> None:
    '''numpy array lags raises TypeError.'''
    with pytest.raises( TypeError, match = "Lags must be an int or a sequence" ):
      CutY( tor.tensor( [ 1, 2 ] ), np.array( [ 1, 2 ] ) )

  def test_lags_set_raises( self ) -> None:
    '''Set lags raises TypeError.'''
    with pytest.raises( TypeError, match = "Lags must be an int or a sequence" ):
      CutY( tor.tensor( [ 1, 2 ] ), { 1, 2 } )

  # ------------------------------------------------------------------
  # Invalid Lags – bad element inside (flat or nested)
  # ------------------------------------------------------------------
  def test_negative_lag_int_raises( self ) -> None:
    '''Negative int lag raises ValueError.'''
    with pytest.raises( ValueError, match = "non-negative" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), -5 )

  def test_negative_lag_in_flat_sequence_raises( self ) -> None:
    '''Negative lag in flat sequence raises ValueError.'''
    with pytest.raises( ValueError, match = "non-negative" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, -2 ] )

  def test_negative_lag_in_nested_raises( self ) -> None:
    '''Negative lag in nested sequence raises ValueError.'''
    with pytest.raises( ValueError, match = "non-negative" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, [ 2, -3 ] ] )

  def test_float_in_flat_sequence_raises( self ) -> None:
    '''Float in flat sequence raises TypeError.'''
    with pytest.raises( TypeError, match = "Each item in Lags must be an int or a flat sequence of ints" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, 2.0 ] )

  def test_float_in_nested_sequence_raises( self ) -> None:
    '''Float in nested sequence raises TypeError.'''
    with pytest.raises( TypeError, match = "All elements inside nested sequence must be integers" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, [ 2, 3.0 ] ] )

  def test_string_in_flat_sequence_raises( self ) -> None:
    '''String in flat sequence raises TypeError.'''
    with pytest.raises( TypeError, match = "Each item in Lags must be an int or a flat sequence of ints" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, "2" ] )

  def test_string_in_nested_sequence_raises( self ) -> None:
    '''String in nested sequence raises TypeError.'''
    with pytest.raises( TypeError, match = "All elements inside nested sequence must be integers" ):
      CutY( tor.tensor( [ 1, 2, 3 ] ), [ 1, [ 2, "3" ] ] )

  # ------------------------------------------------------------------
  # Prove that common sequence types work (no false positive from else)
  # ------------------------------------------------------------------
  def test_flat_list_works( self ) -> None:
    '''List lags work correctly.'''
    y = tor.arange( 10 )
    result = CutY( y, [ 2, 4, 6 ] )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 6 : ] )

  def test_flat_tuple_works( self ) -> None:
    '''Tuple lags work correctly.'''
    y = tor.arange( 10 )
    result = CutY( y, ( 3, 5 ) )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 5 : ] )

  def test_range_works( self ) -> None:
    '''Range lags work correctly.'''
    y = tor.arange( 10 )
    result = CutY( y, range( 2, 5 ) )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 4 : ] )

  # ------------------------------------------------------------------
  # Additional edge cases & device / type preservation
  # ------------------------------------------------------------------
  def test_lag_zero_int_returns_original( self ) -> None:
    '''Int lag of 0 returns the full tensor unchanged.'''
    y = tor.tensor( [ 1, 2, 3, 4, 5 ] )
    result = CutY( y, 0 )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert result.shape == y.shape
    assert tor.equal( result, y )

  def test_preserves_device( self ) -> None:
    '''Output is on the same device as the input.'''
    y = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    result = CutY( y, 1 )
    assert result.device == y.device

  def test_lags_bytes_raises( self ) -> None:
    '''Bytes lags raises TypeError (bytes is explicitly excluded).'''
    with pytest.raises( TypeError, match = "Lags must be an int or a sequence" ):
      CutY( tor.tensor( [ 1, 2 ] ), b"abc" )

  def test_nested_single_list_works( self ) -> None:
    '''Nested single list of ints works correctly.'''
    y = tor.arange( 10 )
    result = CutY( y, ( [ 2, 5 ], ) )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 5 : ] )

  def test_nested_range_works( self ) -> None:
    '''Range inside a nested sequence works correctly.'''
    y = tor.arange( 10 )
    result = CutY( y, [ range( 2, 5 ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 4 : ] )

  def test_mixed_ints_and_nested_lists( self ) -> None:
    '''Mixed ints and nested lists produces correct slice.'''
    y = tor.arange( 10 )
    result = CutY( y, ( 2, [ 3, 5 ], 1 ) )
    assert isinstance( result, tor.Tensor )
    assert result.dtype == y.dtype
    assert result.ndim == 1
    assert tor.equal( result, y[ 5 : ] )

  # ------------------------------------------------------------------
  # Empty top‑level sequences are now forbidden (raises ValueError)
  # ------------------------------------------------------------------
  def test_empty_tuple_raises( self ) -> None:
    '''Empty tuple lags raises ValueError.'''
    y = tor.tensor( [ 9, 8, 7 ] )
    with pytest.raises( ValueError, match = "Empty Lags is not allowed" ):
      CutY( y, () )

  def test_empty_list_raises( self ) -> None:
    '''Empty list lags raises ValueError.'''
    y = tor.tensor( [ 1.0, 2.0 ] )
    with pytest.raises( ValueError, match = "Empty Lags is not allowed" ):
      CutY( y, [] )
