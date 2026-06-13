import pytest
import torch as tor
import numpy as np
from typing import Union, List
from NARMAX.CTors import Lagger


# ----- helpers for concise tensor creation -----
def t( *vals: Union[ int, float ] ) -> tor.Tensor:
  '''Create 1D tor float tensor from list of numbers.'''
  return tor.tensor( vals, dtype = tor.get_default_dtype() )


# =====================================================================
# Section 1: Validation & error paths
# =====================================================================

def test_empty_data_raises_valueerror() -> None:
  '''Empty data raises ValueError.'''
  with pytest.raises( ValueError, match = "Data must contain at least one" ) as exc_info:
    Lagger( [], [] )
  assert "regressor Matrix" in str( exc_info.value )
  assert "Data" in str( exc_info.value )


def test_data_lags_length_mismatch() -> None:
  '''Data/lags length mismatch raises AssertionError.'''
  with pytest.raises( AssertionError, match = "don't correspond" ):
    Lagger( [ t( 1, 2 ) ], [ 0, 0 ] )


def test_data_regnames_length_mismatch() -> None:
  '''Data/RegNames length mismatch raises AssertionError.'''
  with pytest.raises( AssertionError, match = "don't correspond" ):
    Lagger( [ t( 1, 2 ) ], [ 0 ], RegNames = [ "x", "y" ] )


def test_inf_in_data_raises() -> None:
  '''Inf in data raises AssertionError.'''
  data = [ t( 1, float( 'inf' ), 3 ) ]
  with pytest.raises( AssertionError, match = "contains inf or nans" ) as exc_info:
    Lagger( data, [ 0 ] )
  assert "Data[ 0 ]" in str( exc_info.value )


def test_nan_in_data_raises() -> None:
  '''NaN in data raises AssertionError.'''
  data = [ t( 1, float( 'nan' ), 3 ) ]
  with pytest.raises( AssertionError, match = "contains inf or nans" ) as exc_info:
    Lagger( data, [ 0 ] )
  assert "Data[ 0 ]" in str( exc_info.value )


def test_regressors_varying_length() -> None:
  '''Varying-length regressors raise AssertionError.'''
  with pytest.raises( AssertionError, match = "same lenght" ):
    Lagger( [ t( 1, 2, 3 ), t( 4, 5 ) ], [ 0, 0 ] )


def test_negative_integer_lag() -> None:
  '''Negative integer lag raises AssertionError.'''
  with pytest.raises( AssertionError, match = "must be >= 0" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ -1 ] )
  assert "Integer Lags elements" in str( exc_info.value )


def test_non_integer_in_sequence_lag() -> None:
  '''Non-integer in sequence lag raises AssertionError.'''
  with pytest.raises( AssertionError, match = "must be integers" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ [ 0, 1.5 ] ] )
  assert "Lags-sublist elements must be integers" in str( exc_info.value )
  assert "Lags[0][1]" in str( exc_info.value )


def test_negative_in_sequence_lag() -> None:
  '''Negative value in sequence lag raises AssertionError.'''
  with pytest.raises( AssertionError, match = "integers >= 0" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ [ 0, -1 ] ] )
  assert "Lags-sublist elements must be integers >= 0" in str( exc_info.value )
  assert "Lags[0][1]" in str( exc_info.value )


def test_invalid_lag_type() -> None:
  '''Invalid lag type raises AssertionError.'''
  with pytest.raises( AssertionError, match = "must be integers or Lists" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ "a" ] )
  assert "All MaxLags elements must be integers or Lists" in str( exc_info.value )
  assert "Lags[0]" in str( exc_info.value )


def test_empty_sequence_lag_raises_valueerror() -> None:
  '''Empty sequence lag raises ValueError.'''
  with pytest.raises( ValueError, match = "must not be an empty sequence" ):
    Lagger( [ t( 1, 2, 3 ) ], [ [] ] )


def test_no_regnames_and_length_gt3_raises() -> None:
  '''No RegNames with >3 regressors raises AssertionError.'''
  data = [ t( 1 ) ] * 4
  with pytest.raises( AssertionError, match = "can't be assumed" ) as exc_info:
    Lagger( data, [ 0 ] * 4 )
  assert "No regressor names were passed" in str( exc_info.value )
  assert "len( Data ) > 3" in str( exc_info.value )


def test_duplicate_lags_raises_valueerror() -> None:
  '''Duplicate lag values raise ValueError.'''
  data = [ t( 1, 2, 3, 4 ), t( 10, 20, 30, 40 ) ]
  with pytest.raises( ValueError, match = "Duplicate lag values" ):
    Lagger( data, [ 2, [ 0, 0 ] ], RegNames = [ "x", "y" ] )


# =====================================================================
# Section 2: Correctness of outputs – basic scenarios
# =====================================================================

def test_integer_lags_two_regressors() -> None:
  '''Integer lags with two regressors produce correct output.'''
  # x = [1,2,3,4,5,6], y = [10,20,30,40,50,60], lags [1,2] => q=2
  x = t( 1, 2, 3, 4, 5, 6 )
  y_data = t( 10, 20, 30, 40, 50, 60 )
  y, regmat, names = Lagger( [ x, y_data ], [ 1, 2 ], RegNames = [ "x", "y" ] )

  # y[k] separated, length = p - q = 4
  expected_y = t( 30, 40, 50, 60 ) # y_data[2:6]
  assert tor.equal( y, expected_y )

  # RegMat: x[k], x[k-1], y[k-1], y[k-2]   (y[k] popped)
  # p=6, q=2, slices:
  # x[k]   : x[2:6] -> [3,4,5,6]
  # x[k-1] : x[1:5] -> [2,3,4,5]
  # y[k-1] : y[1:5] -> [20,30,40,50]
  # y[k-2] : y[0:4] -> [10,20,30,40]
  expected_regmat = tor.tensor( [
        [ 3., 2., 20., 10. ],
        [ 4., 3., 30., 20. ],
        [ 5., 4., 40., 30. ],
        [ 6., 5., 50., 40. ]
    ], dtype = tor.get_default_dtype() )
  assert regmat.shape == ( 4, 4 )
  assert tor.allclose( regmat, expected_regmat )

  expected_names = np.array( [ "x[k]", "x[k-1]", "y[k-1]", "y[k-2]" ], dtype = np.str_ )
  assert np.array_equal( names, expected_names )


def test_sequence_lags_without_y_zero() -> None:
  '''Sequence lags excluding 0 produce correct y.'''
  data = [ t( 1, 2, 3, 4, 5 ), t( 10, 20, 30, 40, 50 ) ]
  lag_spec = [ 1, [ 1, 2 ] ] # x:0,1 ; y:1,2 => q=2
  y, regmat, names = Lagger( data, lag_spec, RegNames = [ "x", "y" ] )

  # y should be Data[1][2:5] = [30,40,50]
  expected_y = t( 30, 40, 50 )
  assert tor.equal( y, expected_y )

  # RegMat: x[k], x[k-1], y[k-1], y[k-2]
  # x[k]: x[2:5]=[3,4,5]; x[k-1]: x[1:4]=[2,3,4]
  # y[k-1]: y[1:4]=[20,30,40]; y[k-2]: y[0:3]=[10,20,30]
  expected_regmat = tor.tensor( [
        [ 3., 2., 20., 10. ],
        [ 4., 3., 30., 20. ],
        [ 5., 4., 40., 30. ]
    ], dtype = tor.get_default_dtype() )
  assert regmat.shape == ( 3, 4 )
  assert tor.allclose( regmat, expected_regmat )
  expected_names = np.array( [ "x[k]", "x[k-1]", "y[k-1]", "y[k-2]" ], dtype = np.str_ )
  assert np.array_equal( names, expected_names )


def test_no_y_regressor() -> None:
  '''No y regressor: y is None, all lags in RegMat.'''
  u = t( 1, 2, 3, 4 )
  v = t( 5, 6, 7, 8 )
  y, regmat, names = Lagger( [ u, v ], [ 0, 0 ], RegNames = [ "u", "v" ] )
  assert y is None
  # two columns: u[k], v[k]
  assert regmat.shape == ( 4, 2 )
  assert regmat.dtype == tor.get_default_dtype()
  assert tor.allclose( regmat[ :, 0 ], u )
  assert tor.allclose( regmat[ :, 1 ], v )
  assert names.dtype.kind == 'U'
  assert np.array_equal( names, np.array( [ "u[k]", "v[k]" ], dtype = np.str_ ) )


def test_default_regnames_length1() -> None:
  '''Default RegNames with length 1 works.'''
  y, regmat, names = Lagger( [ t( 1, 2 ) ], [ 0 ] )
  assert y is None # length 1 -> only "x", no y
  assert regmat.shape == ( 2, 1 )
  assert regmat.dtype == tor.get_default_dtype()
  assert names.dtype.kind == 'U'
  assert np.array_equal( names, np.array( [ "x[k]" ] ) )


def test_default_regnames_length2() -> None:
  '''Default RegNames with length 2 works.'''
  y, regmat, names = Lagger( [ t( 1, 2 ), t( 3, 4 ) ], [ 0, 0 ] )
  # length 2 => ["x","y"] -> y[k] separated
  assert y is not None
  assert isinstance( y, tor.Tensor )
  assert y.dtype == tor.get_default_dtype()
  assert tor.equal( y, t( 3, 4 ) ) # no trimming because q=0
  assert regmat.shape == ( 2, 1 ) # only x[k] stays
  assert regmat.dtype == tor.get_default_dtype()
  assert names.dtype.kind == 'U'
  assert np.array_equal( names, np.array( [ "x[k]" ] ) )


def test_default_regnames_length3() -> None:
  '''Default RegNames with length 3 works.'''
  e = t( 100, 200, 300 )
  y, regmat, names = Lagger( [ t( 1, 2, 3 ), t( 4, 5, 6 ), e ], [ 0, 0, 0 ] )
  # length 3 => ["x","y","e"] -> y[k] separated, e[k] stays
  assert y is not None
  assert isinstance( y, tor.Tensor )
  assert y.dtype == tor.get_default_dtype()
  assert tor.equal( y, t( 4, 5, 6 ) )
  assert regmat.shape == ( 3, 2 ) # x[k], e[k]
  assert regmat.dtype == tor.get_default_dtype()
  assert tor.allclose( regmat[ :, 0 ], t( 1, 2, 3 ) )
  assert tor.allclose( regmat[ :, 1 ], e )
  assert names.dtype.kind == 'U'
  assert np.array_equal( names, np.array( [ "x[k]", "e[k]" ] ) )


# =====================================================================
# Section 3: Edge cases and new robustness
# =====================================================================

def test_tuple_lags() -> None:
  '''Tuple is accepted as sequence of lags.'''
  x = t( 1, 2, 3, 4 )
  y_data = t( 10, 20, 30, 40 )
  # use tuple (0,2) for y -> lags 0 and 2 for y, integer 1 for x => q=2
  y, regmat, names = Lagger( [ x, y_data ], [ 1, ( 0, 2 ) ], RegNames = [ "x", "y" ] )
  # y[k] present, so y = y_data[2:4] = [30,40]
  assert tor.equal( y, t( 30, 40 ) )
  # x[k], x[k-1], y[k-2] (y[k] popped)
  # x[k]   : x[2:4] = [3,4]
  # x[k-1] : x[1:3] = [2,3]
  # y[k-2] : y[0:2] = [10,20]
  expected = tor.tensor( [
        [ 3., 2., 10. ],
        [ 4., 3., 20. ]
    ], dtype = tor.get_default_dtype() )
  assert tor.allclose( regmat, expected )
  assert np.array_equal( names, np.array( [ "x[k]", "x[k-1]", "y[k-2]" ] ) )


def test_empty_regmat_after_y_removal() -> None:
  '''Only y present: RegMat becomes empty after y removal.'''
  data = [ t( 1, 2, 3 ) ]
  lag_spec = [ [ 0 ] ]
  regnames = [ "y" ]
  y, regmat, names = Lagger( data, lag_spec, RegNames = regnames )
  assert y is not None
  assert tor.equal( y, t( 1, 2, 3 ) )
  # RegMat empty, shape (3,0)
  assert regmat.shape == ( 3, 0 )
  # names empty array of strings
  assert names.dtype == np.dtype( '<U1' ) or names.dtype.kind == 'U'
  assert len( names ) == 0


def test_lag_equals_or_exceeds_p_raises_valueerror() -> None:
  '''Lag >= data length raises ValueError.'''
  # q == p
  with pytest.raises( ValueError, match = "q = 3.*p = 3" ):
    Lagger( [ t( 1, 2, 3 ) ], [ 3 ], RegNames = [ "x" ] )
  # q > p
  with pytest.raises( ValueError, match = "q = 5.*p = 3" ):
    Lagger( [ t( 1, 2, 3 ) ], [ 5 ], RegNames = [ "x" ] )


def test_zero_length_tensors() -> None:
  '''Zero-length tensors raise ValueError.'''
  data = [ tor.empty( 0, dtype = tor.get_default_dtype() ), tor.empty( 0, dtype = tor.get_default_dtype() ) ]
  with pytest.raises( ValueError, match = "Not enough Data to create desired lags" ):
    Lagger( data, [ 0, 0 ], RegNames = [ "x", "y" ] )


def test_large_lag_exceeds_data_length() -> None:
  '''Lag exceeding data length raises ValueError.'''
  data = [ t( 1, 2, 3 ) ] # p=3
  with pytest.raises( ValueError, match = "Not enough Data to create desired lags" ) as exc_info:
    Lagger( data, [ 5 ], RegNames = [ "x" ] )
  assert "q = 5" in str( exc_info.value )
  assert "p = 3" in str( exc_info.value )


def test_regnames_as_numpy_array() -> None:
  '''RegNames can be a numpy array of str.'''
  data = [ t( 1, 2, 3, 4 ), t( 10, 20, 30, 40 ) ]
  regnames = np.array( [ "a", "b" ] )
  y, regmat, names = Lagger( data, [ 0, 0 ], RegNames = regnames )
  # no y -> y=None
  assert y is None
  # both columns stay
  assert regmat.shape == ( 4, 2 )
  assert np.array_equal( names, np.array( [ "a[k]", "b[k]" ], dtype = np.str_ ) )


def test_no_q_trim_when_q_zero() -> None:
  '''All lags zero: no trimming.'''
  x = t( 5, 6, 7 )
  y_data = t( 1, 2, 3 )
  y, regmat, names = Lagger( [ x, y_data ], [ 0, 0 ], RegNames = [ "x", "y" ] )
  # y separated: y_data[0:3] = whole
  assert tor.equal( y, y_data )
  assert y.dtype == tor.get_default_dtype()
  # regmat = x[k] only
  assert tor.equal( regmat.view( -1 ), x )
  assert regmat.dtype == tor.get_default_dtype()
  assert names.dtype.kind == 'U'
  assert names.tolist() == [ "x[k]" ]


def test_multiple_regressors_no_y() -> None:
  '''Multiple regressors with mixed lags, no y.'''
  a = t( 1, 2, 3, 4, 5 )
  b = t( 10, 20, 30, 40, 50 )
  c = t( 100, 200, 300, 400, 500 )
  lags = [ 2, [ 1, 3 ], 0 ] # q = max(2,3,0)=3
  y, regmat, names = Lagger( [ a, b, c ], lags, RegNames = [ "a", "b", "c" ] )
  # p=5, q=3 => length 2 rows
  # a: lags 0,1,2 -> a[k], a[k-1], a[k-2]
  # b: lags 1,3   -> b[k-1], b[k-3]
  # c: lag 0      -> c[k]
  # Expected slices:
  # a[k]   : a[3:5] = [4,5]
  # a[k-1] : a[2:4] = [3,4]
  # a[k-2] : a[1:3] = [2,3]
  # b[k-1] : b[2:4] = [30,40]
  # b[k-3] : b[0:2] = [10,20]
  # c[k]   : c[3:5] = [400,500]
  expected = tor.tensor( [
        [ 4., 3., 2., 30., 10., 400. ],
        [ 5., 4., 3., 40., 20., 500. ]
    ], dtype = tor.get_default_dtype() )
  assert tor.allclose( regmat, expected )
  assert names.tolist() == [
        "a[k]", "a[k-1]", "a[k-2]",
        "b[k-1]", "b[k-3]",
        "c[k]"
    ]
  assert y is None


def test_duplicate_lags_not_triggered_when_different_regressors() -> None:
  '''Same lag values for different regressors is allowed.'''
  x = t( 1, 2, 3, 4 )
  y_data = t( 10, 20, 30, 40 )
  # both have lags [0,1]
  y, regmat, _ = Lagger( [ x, y_data ], [ [ 0, 1 ], [ 0, 1 ] ], RegNames = [ "x", "y" ] )
  # y[k] separated, x[k],x[k-1], y[k-1] remain
  # p=4, q=1 => rows 3
  # x[k]: x[1:4]=[2,3,4]; x[k-1]: x[0:3]=[1,2,3]
  # y[k-1]: y[0:3]=[10,20,30]
  expected = tor.tensor( [
        [ 2., 1., 10. ],
        [ 3., 2., 20. ],
        [ 4., 3., 30. ]
    ], dtype = tor.get_default_dtype() )
  assert tor.allclose( regmat, expected )
  # y separated = y[1:4] = [20,30,40]
  assert tor.equal( y, t( 20, 30, 40 ) )


def test_float_lag_raises() -> None:
  '''Float as top-level lag raises AssertionError.'''
  with pytest.raises( AssertionError, match = "must be integers or Lists" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ 0.5 ], RegNames = [ "x" ] )
  assert "Lags[0]" in str( exc_info.value )


def test_float_in_sequence_lag_raises() -> None:
  '''Float in sequence lag raises AssertionError.'''
  with pytest.raises( AssertionError, match = "must be integers" ) as exc_info:
    Lagger( [ t( 1, 2, 3 ) ], [ [ 0, 1.5 ] ], RegNames = [ "x" ] )
  assert "Lags[0][1]" in str( exc_info.value )


def test_numpy_int_lag_rejected() -> None:
  '''Numpy integer lags raise AssertionError (not isinstance of Python int).'''
  import numpy as np
  x = t( 1, 2, 3, 4, 5 )
  y_data = t( 10, 20, 30, 40, 50 )
  with pytest.raises( AssertionError, match = "must be integers or Lists" ) as exc_info:
    Lagger( [ x, y_data ], [ np.int64( 1 ), np.int64( 2 ) ], RegNames = [ "x", "y" ] )
  assert "Lags[0]" in str( exc_info.value )


def test_data_as_tuple() -> None:
  '''Data passed as tuple of tensors works.'''
  x = t( 1, 2, 3, 4 )
  y_data = t( 10, 20, 30, 40 )
  data_tuple = ( x, y_data )
  y, regmat, names = Lagger( data_tuple, [ 0, 0 ], RegNames = [ "x", "y" ] )
  assert y is not None
  assert tor.equal( y, y_data )
  assert regmat.shape == ( 4, 1 )
  assert tor.equal( regmat.view( -1 ), x )
  assert names.tolist() == [ "x[k]" ]


def test_y_first_regressor_no_zero_lag() -> None:
  '''"y" as first regressor with lags not including 0 uses fallback.'''
  y_data = t( 10, 20, 30, 40, 50 )
  x = t( 1, 2, 3, 4, 5 )
  # y lags [1, 2] (no 0), x lags [0, 1] => q=2, p=5 => 3 rows
  y, regmat, names = Lagger( [ y_data, x ], [ [ 1, 2 ], [ 0, 1 ] ], RegNames = [ "y", "x" ] )
  # y[k] not in OutNames, so fallback: Data[0][q:p] = y_data[2:5] = [30,40,50]
  assert y is not None
  assert tor.equal( y, t( 30, 40, 50 ) )
  assert y.dtype == tor.get_default_dtype()
  # y[k-1], y[k-2], x[k], x[k-1]
  # y[k-1]: y[1:4]=[20,30,40]
  # y[k-2]: y[0:3]=[10,20,30]
  # x[k]: x[2:5]=[3,4,5]
  # x[k-1]: x[1:4]=[2,3,4]
  expected = tor.tensor( [
        [ 20., 10., 3., 2. ],
        [ 30., 20., 4., 3. ],
        [ 40., 30., 5., 4. ]
    ], dtype = tor.get_default_dtype() )
  assert tor.allclose( regmat, expected )
  expected_names = np.array( [ "y[k-1]", "y[k-2]", "x[k]", "x[k-1]" ], dtype = np.str_ )
  assert np.array_equal( names, expected_names )
  assert names.dtype.kind == 'U'


def test_no_y_in_regnames() -> None:
  '''No "y" in RegNames: y is None, all columns kept.'''
  a = t( 1, 2, 3, 4 )
  b = t( 10, 20, 30, 40 )
  c = t( 100, 200, 300, 400 )
  y, regmat, names = Lagger( [ a, b, c ], [ 0, 1, 2 ], RegNames = [ "a", "b", "c" ] )
  assert y is None
  # q = 2, p = 4 => rows 2
  # a: Lags[0]=0 (int) -> LagList=[0]        -> a[k]
  # b: Lags[1]=1 (int) -> LagList=[0,1]      -> b[k], b[k-1]
  # c: Lags[2]=2 (int) -> LagList=[0,1,2]    -> c[k], c[k-1], c[k-2]
  # Total columns = 1 + 2 + 3 = 6
  assert regmat.shape == ( 2, 6 )
  assert regmat.dtype == tor.get_default_dtype()
  expected = tor.tensor( [
        [ 3., 30., 20., 300., 200., 100. ],
        [ 4., 40., 30., 400., 300., 200. ]
    ], dtype = tor.get_default_dtype() )
  assert tor.allclose( regmat, expected )
  assert names.tolist() == [ "a[k]", "b[k]", "b[k-1]", "c[k]", "c[k-1]", "c[k-2]" ]
  assert names.dtype.kind == 'U'


# =====================================================================
# Section 4: dtype and shape invariants
# =====================================================================
def test_output_dtypes() -> None:
  '''Output dtypes are correct.'''
  data = [ t( 1, 2, 3 ), t( 4, 5, 6 ) ]
  y, regmat, names = Lagger( data, [ 0, 0 ], RegNames = [ "x", "y" ] )
  assert y.dtype == tor.get_default_dtype()
  assert regmat.dtype == tor.get_default_dtype()
  assert np.issubdtype( names.dtype, np.str_ )
  # Also check when y is None (no "y" regressor)
  y2, regmat2, names2 = Lagger( [ t( 1, 2 ) ], [ 0 ], RegNames = [ "x" ] )
  assert y2 is None
  assert regmat2.dtype == tor.get_default_dtype()
  assert np.issubdtype( names2.dtype, np.str_ )


def test_outnames_are_string_array_when_empty() -> None:
  '''Empty names array has string dtype.'''
  # case where only y[k] exists and is popped => names empty
  y, regmat, names = Lagger( [ t( 1, 2, 3 ) ], [ [ 0 ] ], RegNames = [ "y" ] )
  assert names.size == 0
  assert names.dtype.kind == 'U' # string/unicode
