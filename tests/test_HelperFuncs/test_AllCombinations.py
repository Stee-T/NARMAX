import numpy as np
from numpy.typing import NDArray
import pytest

from NARMAX.HelperFuncs import AllCombinations

class TestAllCombinations:
  '''Test suite for the AllCombinations helper function.'''

  # --- fixtures ---
  @pytest.fixture
  def dtype_i64( self ) -> type: return np.int64

  @pytest.fixture
  def dtype_i32( self ) -> type: return np.int32

  # --- normal behaviour ---
  def test_basic_functionality( self, dtype_i64 ) -> None:
    '''Basic combination with imposed and input sequence.'''
    imposed = np.array( [ 10, 20 ], dtype = np.int64 )
    seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 3, 3 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert np.array_equal( result[ :, : -1 ], np.tile( imposed, ( 3, 1 ) ) )
    assert np.array_equal( result[ :, -1 ], seq )

  def test_removes_imposed_from_inputseq( self, dtype_i64 ) -> None:
    '''Imposed values removed from the output sequence column.'''
    imposed = np.array( [ 2, 4 ], dtype = np.int64 )
    seq = np.array( [ 1, 2, 3, 4, 5 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    expected_new = np.array( [ 1, 3, 5 ], dtype = np.int64 )
    assert result.shape == ( 3, 3 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert np.array_equal( result[ :, -1 ], expected_new )
    assert not np.any( np.isin( result[ :, -1 ], imposed ) )

  def test_preserves_inputseq_order( self, dtype_i64 ) -> None:
    '''Input sequence order is preserved in output.'''
    imposed = np.array( [ 100 ], dtype = np.int64 )
    seq = np.array( [ 5, 3, 1, 7, 9 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 5, 2 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert np.array_equal( result[ :, -1 ], seq )

  # --- edge cases (empty, all filtered) ---
  def test_empty_imposed( self, dtype_i64 ) -> None:
    '''Empty imposed array returns identity mapping.'''
    imposed = np.array( [], dtype = np.int64 )
    seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 3, 1 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert np.array_equal( result[ :, 0 ], seq )

  def test_empty_result( self, dtype_i64 ) -> None:
    '''All values imposed yields empty result.'''
    imposed = np.array( [ 1, 2, 3 ], dtype = np.int64 )
    seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 0, 4 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert result.flags[ "C_CONTIGUOUS" ]

  def test_empty_inputseq( self, dtype_i64 ) -> None:
    '''Empty input sequence yields empty result.'''
    imposed = np.array( [ 5, 6 ], dtype = np.int64 )
    seq = np.array( [], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 0, 3 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64

  def test_both_empty( self, dtype_i64 ) -> None:
    '''Both arrays empty yields empty result with single column.'''
    imposed = np.array( [], dtype = np.int64 )
    seq = np.array( [], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 0, 1 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    assert result.flags[ "C_CONTIGUOUS" ]

  # --- guard: duplicate InputSeq raises ValueError ---
  def test_duplicate_in_inputseq_raises( self, dtype_i64 ) -> None:
    '''Duplicate values in InputSeq raises ValueError.'''
    imposed = np.array( [ 10 ], dtype = np.int64 )
    seq = np.array( [ 1, 1, 2, 3 ], dtype = np.int64 ) # duplicate

    with pytest.raises( ValueError, match = "InputSeq must not contain duplicate values" ):
      AllCombinations( imposed, seq, dtype_i64 )

  def test_inputseq_with_duplicate_that_is_also_imposed_raises( self, dtype_i64 ) -> None:
    '''Duplicate check runs before setdiff1d, catches matches in imposed.'''
    imposed = np.array( [ 2 ], dtype = np.int64 )
    seq = np.array( [ 1, 2, 2, 3 ], dtype = np.int64 ) # duplicate 2

    with pytest.raises( ValueError, match = "InputSeq must not contain duplicate values" ):
      AllCombinations( imposed, seq, dtype_i64 )

  # --- guard: unsafe INT_TYPE raises ValueError ---
  def test_int32_type_with_int64_values_raises( self, dtype_i32 ) -> None:
    '''int32 cannot safely hold int64 values, raises ValueError.'''
    imposed = np.array( [ 10 ], dtype = np.int64 )
    seq = np.array( [ 100 ], dtype = np.int64 )

    # int32 cannot safely hold int64 values.
    with pytest.raises( ValueError, match = "cannot safely hold" ):
      AllCombinations( imposed, seq, dtype_i32 )

  def test_int64_type_is_allowed( self, dtype_i64 ) -> None:
    '''int64 type is accepted without error.'''
    imposed = np.array( [ np.iinfo( np.int64 ).max ], dtype = np.int64 )
    seq = np.array( [ 0 ], dtype = np.int64 )

    # Should not raise.
    result = AllCombinations( imposed, seq, dtype_i64 )
    assert result.dtype == dtype_i64
    assert result.ndim == 2
    assert result.shape == ( 1, 2 )
    assert result.flags[ "C_CONTIGUOUS" ]
    assert np.array_equal( result, np.array( [ [ np.iinfo( np.int64 ).max, 0 ] ], dtype = np.int64 ) )

  def test_int32_type_with_int32_values_is_allowed( self, dtype_i32 ) -> None:
    '''int32 type is accepted when inputs are also int32.'''
    imposed = np.array( [ 1, 2 ], dtype = np.int32 )
    seq = np.array( [ 3, 4 ], dtype = np.int32 )

    result = AllCombinations( imposed, seq, dtype_i32 )

    assert result.dtype == dtype_i32
    assert result.ndim == 2
    assert result.shape == ( 2, 3 )
    assert np.array_equal( result[ :, : -1 ], np.tile( imposed, ( 2, 1 ) ) )
    assert np.array_equal( result[ :, -1 ], seq )

  # --- ImposedRegs with duplicates (allowed) ---
  def test_imposed_duplicates_are_preserved( self, dtype_i64 ) -> None:
    '''Duplicate values in ImposedRegs are preserved in output.'''
    imposed = np.array( [ 7, 7, 8 ], dtype = np.int64 )
    seq = np.array( [ 1, 2 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.shape == ( 2, 4 )
    assert result.ndim == 2
    assert result.dtype == dtype_i64
    expected_row_start = np.array( [ 7, 7, 8 ], dtype = np.int64 )
    assert np.array_equal( result[ 0, : -1 ], expected_row_start )
    assert np.array_equal( result[ 1, : -1 ], expected_row_start )
    assert np.array_equal( result[ :, -1 ], seq )

  # --- misc ---
  def test_result_is_c_contiguous( self, dtype_i64 ) -> None:
    '''Result array is C-contiguous.'''
    imposed = np.array( [ 1 ], dtype = np.int64 )
    seq = np.array( [ 10, 20 ], dtype = np.int64 )

    result = AllCombinations( imposed, seq, dtype_i64 )

    assert result.flags[ "C_CONTIGUOUS" ]
    assert result.flags[ "WRITEABLE" ]
    assert result.dtype == dtype_i64
    assert result.shape == ( 2, 2 )
    assert result.ndim == 2
    assert np.array_equal( result, np.array( [ [ 1, 10 ], [ 1, 20 ] ], dtype = np.int64 ) )
