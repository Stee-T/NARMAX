import torch
import pytest

from NARMAX.HelperFuncs import DeleteColumn


class TestDeleteColumn:
  '''Tests for the updated DeleteColumn function.'''

  # ------------------------------------------------------------------
  # Valid operations
  # ------------------------------------------------------------------
  def test_delete_last_column_2d( self ) -> None:
    '''Delete the last column of a 2D tensor.'''
    t = torch.arange( 9 ).reshape( 3, 3 )
    expected = torch.tensor( [ [ 0, 1 ],
                                 [ 3, 4 ],
                                 [ 6, 7 ] ] )
    result = DeleteColumn( t, 2 )
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.dtype == t.dtype
    assert result.device == t.device

  def test_delete_first_column_2d( self ) -> None:
    '''Delete the first column of a 2D tensor.'''
    t = torch.arange( 9 ).reshape( 3, 3 )
    expected = torch.tensor( [ [ 1, 2 ],
                                 [ 4, 5 ],
                                 [ 7, 8 ] ] )
    result = DeleteColumn( t, 0 )
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.dtype == t.dtype
    assert result.device == t.device

  def test_delete_middle_column_2d( self ) -> None:
    '''Delete a middle column of a 2D tensor.'''
    t = torch.arange( 9 ).reshape( 3, 3 )
    expected = torch.tensor( [ [ 0, 2 ],
                                 [ 3, 5 ],
                                 [ 6, 8 ] ] )
    result = DeleteColumn( t, 1 )
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.dtype == t.dtype
    assert result.device == t.device

  def test_single_column_2d( self ) -> None:
    '''Delete the only column from a 2D tensor.'''
    t = torch.arange( 3 ).reshape( 3, 1 )
    result = DeleteColumn( t, 0 )
    assert isinstance( result, torch.Tensor )
    assert result.shape == ( 3, 0 )
    assert result.numel() == 0
    assert result.dtype == t.dtype
    assert result.device == t.device

  def test_3d_tensor_delete_column( self ) -> None:
    '''Delete a column from a 3D tensor.'''
    # Function works for any tensor with >=2 dimensions.
    t = torch.arange( 24 ).reshape( 2, 3, 4 )
    result = DeleteColumn( t, 1 ) # delete second column
    expected = torch.cat( [ t[ :, : 1, : ], t[ :, 2 :, : ] ], dim = 1 )
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.shape == ( 2, 2, 4 )
    assert result.dtype == t.dtype
    assert result.device == t.device

  def test_3d_tensor_delete_last_column( self ) -> None:
    '''Delete the last column from a 3D tensor.'''
    t = torch.arange( 24 ).reshape( 2, 3, 4 )
    result = DeleteColumn( t, 2 ) # last column (index == shape[1]-1)
    expected = t[ :, : -1, : ]
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.shape == ( 2, 2, 4 )
    assert result.dtype == t.dtype
    assert result.device == t.device

  # ------------------------------------------------------------------
  # View vs copy behaviour
  # ------------------------------------------------------------------
  def test_delete_last_column_returns_view( self ) -> None:
    '''Last-column deletion returns a view sharing storage.'''
    t = torch.arange( 9, dtype = torch.float ).reshape( 3, 3 )
    result = DeleteColumn( t, 2 )
    assert isinstance( result, torch.Tensor )
    assert result.shape == ( 3, 2 )
    assert torch.equal( result, t[ :, : 2 ] )
    # Should share underlying storage with input
    assert result.data_ptr() == t.data_ptr()
    # Verify it is indeed a view by modifying through it
    result[ 0, 0 ] = 99.0
    assert t[ 0, 0 ] == 99.0

  def test_delete_non_last_column_returns_copy( self ) -> None:
    '''Non-last-column deletion returns a copy.'''
    t = torch.arange( 9, dtype = torch.float ).reshape( 3, 3 )
    result = DeleteColumn( t, 0 )
    assert isinstance( result, torch.Tensor )
    assert result.shape == ( 3, 2 )
    assert torch.equal( result, torch.tensor( [ [ 1., 2. ],
                                                  [ 4., 5. ],
                                                  [ 7., 8. ] ] ) )
    # Should be a fresh tensor, not sharing storage
    assert result.data_ptr() != t.data_ptr()
    # Modification must not affect original
    result[ 0, 0 ] = 99.0
    assert t[ 0, 0 ] == 0.0

  # ------------------------------------------------------------------
  # Error cases
  # ------------------------------------------------------------------
  def test_negative_index_raises_index_error( self ) -> None:
    '''Negative column index raises IndexError.'''
    t = torch.arange( 9 ).reshape( 3, 3 )
    with pytest.raises( IndexError, match = r"Column index -1 out of bounds for 3 columns" ):
      DeleteColumn( t, -1 )
    with pytest.raises( IndexError, match = r"Column index -2 out of bounds for 3 columns" ):
      DeleteColumn( t, -2 )
    with pytest.raises( IndexError, match = r"Column index -10 out of bounds for 3 columns" ):
      DeleteColumn( t, -10 )

  def test_out_of_bounds_positive_index_raises( self ) -> None:
    '''Out-of-bounds positive index raises IndexError.'''
    t = torch.arange( 9 ).reshape( 3, 3 )
    with pytest.raises( IndexError, match = r"Column index 3 out of bounds for 3 columns" ):
      DeleteColumn( t, 3 ) # n_cols = 3
    with pytest.raises( IndexError, match = r"Column index 10 out of bounds for 3 columns" ):
      DeleteColumn( t, 10 )

  def test_1d_tensor_raises_value_error( self ) -> None:
    '''1D input raises ValueError.'''
    t = torch.arange( 5 )
    with pytest.raises( ValueError, match = r"at least 2D" ):
      DeleteColumn( t, 0 )

  def test_0d_tensor_raises_value_error( self ) -> None:
    '''0D input raises ValueError.'''
    t = torch.tensor( 42 )
    with pytest.raises( ValueError, match = r"at least 2D" ):
      DeleteColumn( t, 0 )

  def test_2d_tensor_zero_columns_raises( self ) -> None:
    '''Zero-column 2D tensor raises IndexError.'''
    t = torch.empty( 3, 0 )
    # n_cols = 0, so any non-negative index is out of bounds
    with pytest.raises( IndexError, match = r"Column index 0 out of bounds for 0 columns" ):
      DeleteColumn( t, 0 )

  def test_negative_index_on_empty_columns_raises( self ) -> None:
    '''Negative index on empty tensor raises IndexError.'''
    t = torch.empty( 3, 0 )
    with pytest.raises( IndexError, match = r"Column index -1 out of bounds for 0 columns" ):
      DeleteColumn( t, -1 ) # negative index also raises

  # ------------------------------------------------------------------
  # Additional edge cases
  # ------------------------------------------------------------------
  def test_4d_tensor_delete_column( self ) -> None:
    '''Delete a column from a 4D tensor (ndim >= 2).'''
    t = torch.arange( 48 ).reshape( 2, 3, 4, 2 )
    result = DeleteColumn( t, 1 )
    expected = torch.cat( [ t[ :, : 1, :, : ], t[ :, 2 :, :, : ] ], dim = 1 )
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.shape == ( 2, 2, 4, 2 )
    assert result.dtype == t.dtype
    assert result.device == t.device

  @pytest.mark.parametrize( "dtype", [ torch.float32, torch.float64, torch.int32, torch.int64 ] )
  def test_delete_column_preserves_dtype( self, dtype ) -> None:
    '''DeleteColumn preserves the input dtype for various dtypes.'''
    t = torch.arange( 9, dtype = dtype ).reshape( 3, 3 )
    result_mid = DeleteColumn( t, 1 )
    assert result_mid.dtype == dtype
    assert torch.equal( result_mid, torch.cat( [ t[ :, : 1 ], t[ :, 2 : ] ], dim = 1 ) )
    result_last = DeleteColumn( t, 2 )
    assert result_last.dtype == dtype
    assert torch.equal( result_last, t[ :, : 2 ] )

  def test_4d_tensor_delete_last_column( self ) -> None:
    '''Delete the last column from a 4D tensor (view path).'''
    t = torch.arange( 48 ).reshape( 2, 3, 4, 2 )
    result = DeleteColumn( t, 2 )
    expected = t[ :, : -1, :, : ]
    assert isinstance( result, torch.Tensor )
    assert torch.equal( result, expected )
    assert result.shape == ( 2, 2, 4, 2 )
    assert result.dtype == t.dtype
    assert result.device == t.device
