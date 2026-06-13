import numpy as np
import pytest
import torch

from NARMAX.HelperFuncs import RemoveDuplicates

# ------------------------------------------------------------------ #
#  Helper / fixtures                                                  #
# ------------------------------------------------------------------ #
@pytest.fixture
def sample_data_no_duplicates() -> tuple:
  '''3 samples, 4 distinct columns.'''
  mat = torch.tensor(
        [ [ 1.0, 2.0, 3.0, 4.0 ],
              [ 5.0, 6.0, 7.0, 8.0 ],
              [ 9.0, 0.0, 1.0, 2.0 ] ],
        dtype = torch.float64,
    )
  names = np.array( [ "a", "b", "c", "d" ], dtype = np.str_ )
  return mat, names


@pytest.fixture
def sample_data_with_duplicates() -> tuple:
  '''Columns 1 and 3 are identical, column 2 is unique, column 0 is unique.'''
  mat = torch.tensor(
        [ [ 1.0, 2.0, 3.0, 2.0 ],
              [ 5.0, 6.0, 7.0, 6.0 ],
              [ 9.0, 0.0, 1.0, 0.0 ] ],
        dtype = torch.float32,
    )
  names = np.array( [ "x", "y", "z", "y_dup" ], dtype = np.str_ )
  return mat, names


@pytest.fixture
def sample_data_all_same() -> tuple:
  '''All columns identical.'''
  mat = torch.full( ( 4, 5 ), 7.0, dtype = torch.float64 )
  names = np.array( [ "c1", "c2", "c3", "c4", "c5" ], dtype = np.str_ )
  return mat, names


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #
class TestRemoveDuplicates:
  # ---------- basic functionality ----------
  def test_no_duplicates( self, sample_data_no_duplicates, capsys ) -> None:
    '''No duplicates: output identical to input.'''
    mat, names = sample_data_no_duplicates
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert torch.equal( out_mat, mat )
    assert out_mat.dtype == mat.dtype
    assert out_mat.device == mat.device
    np.testing.assert_array_equal( out_names, names )
    assert out_names.dtype == names.dtype
    np.testing.assert_array_equal( idx, np.arange( len( names ), dtype = np.int64 ) )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

  def test_with_duplicates( self, sample_data_with_duplicates, capsys ) -> None:
    '''Duplicates are removed, warning printed.'''
    mat, names = sample_data_with_duplicates
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    # expected kept columns: 0, 1, 2 (original indices, order preserved)
    expected_mat = mat[ :, [ 0, 1, 2 ] ]
    expected_names = names[ [ 0, 1, 2 ] ]

    assert torch.equal( out_mat, expected_mat )
    assert out_mat.dtype == mat.dtype
    assert out_mat.device == mat.device
    np.testing.assert_array_equal( out_names, expected_names )
    assert out_names.dtype == names.dtype
    np.testing.assert_array_equal( idx, np.array( [ 0, 1, 2 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "1 redundant regressor" in captured.out
    assert "was eliminated" in captured.out

  def test_all_duplicates( self, sample_data_all_same, capsys ) -> None:
    '''All columns identical: only first kept.'''
    mat, names = sample_data_all_same
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( mat.shape[ 0 ], 1 )
    assert torch.equal( out_mat, mat[ :, [ 0 ] ] )
    assert out_mat.dtype == mat.dtype
    assert out_mat.device == mat.device
    np.testing.assert_array_equal( out_names, np.array( [ "c1" ], dtype = np.str_ ) )
    assert out_names.dtype == names.dtype
    np.testing.assert_array_equal( idx, np.array( [ 0 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "4 redundant regressors were" in captured.out

  # ---------- empty / degenerate inputs ----------
  def test_zero_columns( self ) -> None:
    '''p>0, n=0: empty output.'''
    mat = torch.ones( ( 5, 0 ), dtype = torch.float64 )
    names = np.array( [], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( 5, 0 )
    assert out_mat.dtype == mat.dtype
    assert out_names.shape == ( 0, )
    assert out_names.dtype == names.dtype
    assert idx.shape == ( 0, )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

  def test_zero_rows_zero_columns( self, capsys ) -> None:
    '''n=0, p=0: empty output, no warning.'''
    mat = torch.empty( ( 0, 0 ), dtype = torch.float32 )
    names = np.array( [], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( 0, 0 )
    assert out_mat.dtype == mat.dtype
    assert out_names.shape == ( 0, )
    assert out_names.dtype == names.dtype
    assert idx.shape == ( 0, )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

  def test_zero_rows_with_columns( self, capsys ) -> None:
    '''n=0, p>0: all identical, only first kept.'''
    mat = torch.empty( ( 0, 4 ), dtype = torch.float64 )
    names = np.array( [ "a", "b", "c", "d" ], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( 0, 1 )
    assert out_mat.dtype == mat.dtype
    assert out_names.tolist() == [ "a" ]
    assert out_names.dtype == names.dtype
    np.testing.assert_array_equal( idx, np.array( [ 0 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

  # ---------- additional edge cases ----------
  def test_single_column( self, capsys ) -> None:
    '''Single column: no duplicates possible, output identical.'''
    mat = torch.tensor( [ [ 1.0 ], [ 2.0 ], [ 3.0 ] ], dtype = torch.float64 )
    names = np.array( [ "only" ], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert torch.equal( out_mat, mat )
    assert out_mat.dtype == mat.dtype
    assert out_mat.shape == ( 3, 1 )
    np.testing.assert_array_equal( out_names, names )
    assert out_names.dtype == names.dtype
    np.testing.assert_array_equal( idx, np.array( [ 0 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out

  def test_two_identical_columns( self, capsys ) -> None:
    '''Two identical columns: only first kept.'''
    mat = torch.tensor( [ [ 1.0, 1.0 ], [ 2.0, 2.0 ] ], dtype = torch.float64 )
    names = np.array( [ "a", "b" ], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( 2, 1 )
    assert out_mat.dtype == mat.dtype
    assert torch.equal( out_mat, mat[ :, [ 0 ] ] )
    np.testing.assert_array_equal( out_names, np.array( [ "a" ], dtype = np.str_ ) )
    np.testing.assert_array_equal( idx, np.array( [ 0 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "1 redundant regressor was" in captured.out

  def test_multiple_duplicate_groups( self, capsys ) -> None:
    '''Multiple independent duplicate groups are all removed.'''
    mat = torch.tensor(
        [ [ 1.0, 2.0, 1.0, 3.0, 2.0 ],
              [ 4.0, 5.0, 4.0, 6.0, 5.0 ] ],
        dtype = torch.float64,
    )
    names = np.array( [ "a", "b", "a_dup", "c", "b_dup" ], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.shape == ( 2, 3 )
    assert out_mat.dtype == mat.dtype
    assert torch.equal( out_mat, mat[ :, [ 0, 1, 3 ] ] )
    np.testing.assert_array_equal( out_names, names[ [ 0, 1, 3 ] ] )
    np.testing.assert_array_equal( idx, np.array( [ 0, 1, 3 ], dtype = np.int64 ) )
    assert idx.dtype == np.int64

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "2 redundant regressors were" in captured.out

  # ---------- device handling ----------
  @pytest.mark.skipif( not torch.cuda.is_available(), reason = "CUDA not available" )
  def test_gpu_tensor_stays_on_gpu( self ) -> None:
    '''GPU tensor stays on GPU after processing.'''
    mat = torch.tensor( [ [ 1.0, 1.0, 1.0 ], [ 2.0, 3.0, 3.0 ] ], device = "cuda" )
    names = np.array( [ "dup", "unique" ], dtype = np.str_ )
    out_mat, out_names, idx = RemoveDuplicates( mat, names )

    assert out_mat.device.type == "cuda"
    # values: first column is [1,2] (unique), second duplicate of it removed
    assert torch.equal( out_mat, torch.tensor( [ [ 1.0, 1.0 ], [ 2.0, 3.0 ] ], device = "cuda" ) )

  # TODO: do the same for Mac and others

  def test_cpu_tensor_stays_on_cpu( self ) -> None:
    '''CPU tensor stays on CPU after processing.'''
    mat = torch.tensor( [ [ 1.0, 1.0 ], [ 2.0, 3.0 ] ], dtype = torch.float64, device = "cpu" )
    out_mat, _, _ = RemoveDuplicates( mat, np.array( [ "x", "y" ] ) )
    assert out_mat.device.type == "cpu"

  # ---------- dtype preservation ----------
  def test_dtype_float32_preserved( self ) -> None:
    '''float32 dtype is preserved.'''
    mat = torch.tensor( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ], dtype = torch.float32 )
    names = np.array( [ "f1", "f2" ], dtype = np.str_ )
    out_mat, _, _ = RemoveDuplicates( mat, names )
    assert out_mat.dtype == torch.float32

  def test_dtype_float64_preserved( self ) -> None:
    '''float64 dtype is preserved.'''
    mat = torch.tensor( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ], dtype = torch.float64 )
    names = np.array( [ "d1", "d2" ], dtype = np.str_ )
    out_mat, _, _ = RemoveDuplicates( mat, names )
    assert out_mat.dtype == torch.float64

  # ---------- returned index properties ----------
  def test_DcFilterIdx_sorted_and_int64( self, sample_data_with_duplicates ) -> None:
    '''Index array is int64 and sorted ascending.'''
    out_mat, out_names, idx = RemoveDuplicates( *sample_data_with_duplicates )
    assert idx.dtype == np.int64
    assert len( idx ) == out_mat.shape[ 1 ]
    assert len( idx ) == len( out_names )
    # must be sorted ascending
    np.testing.assert_array_equal( idx, np.sort( idx ) )
    # idx values are the kept column indices in the original matrix
    mat, _ = sample_data_with_duplicates
    assert idx[ -1 ] < mat.shape[ 1 ]

  # ---------- NaN handling (bit‑identical columns) ----------
  def test_raises_on_1d_tensor( self ) -> None:
    '''1D input raises ValueError.'''
    mat = torch.tensor( [ 1.0, 2.0, 3.0 ] )
    names = np.array( [ "a", "b", "c" ], dtype = np.str_ )
    with pytest.raises( ValueError, match = r"RegMat must be 2D \(n_rows, n_cols\)" ):
      RemoveDuplicates( mat, names )

  def test_raises_on_3d_tensor( self ) -> None:
    '''3D input raises ValueError.'''
    mat = torch.zeros( ( 2, 3, 4 ) )
    names = np.array( [ "x" ], dtype = np.str_ )
    with pytest.raises( ValueError, match = r"RegMat must be 2D \(n_rows, n_cols\)" ):
      RemoveDuplicates( mat, names )

  def test_raises_on_nan_values( self ) -> None:
    '''NaN values raise ValueError.'''
    mat = torch.tensor( [ [ 1.0, float( "nan" ) ], [ 2.0, 3.0 ] ] )
    names = np.array( [ "a", "b" ], dtype = np.str_ )
    with pytest.raises( ValueError, match = "RegMat contains NaNs" ):
      RemoveDuplicates( mat, names )

  def test_raises_on_inf_values( self ) -> None:
    '''Inf values raise ValueError.'''
    # torch.isinf catches both +inf and -inf
    mat = torch.tensor( [ [ 1.0, float( "inf" ) ], [ 2.0, float( "-inf" ) ] ] )
    names = np.array( [ "a", "b" ], dtype = np.str_ )
    with pytest.raises( ValueError, match = "RegMat contains Infs" ):
      RemoveDuplicates( mat, names )
