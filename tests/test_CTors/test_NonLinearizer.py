import pytest
import torch
import numpy as np
from typing import Optional, Sequence, Union

from NARMAX import NonLinearity as NonLinearity
from NARMAX.CTors import NonLinearizer

# ---------------------------------------------------------------------------
# Mock NonLinearity class
# ---------------------------------------------------------------------------

id_func = NonLinearity( "id", lambda x: x )
sq_func = NonLinearity( "sq", lambda x: x**2 )
cube_func = NonLinearity( "cube", lambda x: x**3 )


class TestNonLinearizer:

  # Helper for tensor equality
  @staticmethod
  def _assert_tensors_equal( a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6 ) -> None:
    '''Assert two tensors are element-wise close.'''
    assert torch.allclose( a, b, atol = tol ), f"Tensors differ:\n{ a }\n{ b }"

  # ============ Basic functionality ======================================

  def test_basic_transformations( self ) -> None:
    '''Basic transformations with id, sq, cube.'''
    Data = torch.tensor( [ [ 1., 2. ], [ 3., 4. ] ] )
    RegNames = [ "u", "v" ]
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( None, Data, RegNames, Functions, None )

    expected_data = torch.cat( [ Data, Data**2, Data**3 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected_data )

    expected_names = np.array( [ "u", "v", "sq(u)", "sq(v)", "cube(u)", "cube(v)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1, 2, 2 ]

  def test_rational_all_false( self ) -> None:
    '''Rational all False: no rational terms added.'''
    Data = torch.tensor( [ [ 1. ], [ 2. ] ] ) # shape (2,1)
    RegNames = [ "x" ]
    y = torch.tensor( [ 3., 4. ] )
    out_data, out_names, M = NonLinearizer( y, Data, RegNames, [ id_func, sq_func ], [ False, False ] )
    expected = torch.cat( [ Data, Data**2 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    assert list( out_names ) == [ "x", "sq(x)" ]
    assert M == [ 0, 1 ]

  def test_rational_for_sq_only( self ) -> None:
    '''Rational only for sq function.'''
    Data = torch.tensor( [ [ 2., 3. ] ] )
    RegNames = [ "a", "b" ]
    y = torch.tensor( [ -1. ] )
    out_data, out_names, M = NonLinearizer( y, Data, RegNames, [ id_func, sq_func ], [ False, True ] )

    expected = torch.cat( [ Data, Data**2, -y.view( -1, 1 ) * ( Data**2 ) ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    expected_names = np.array( [ "a", "b", "sq(a)", "sq(b)", "~/sq(a)", "~/sq(b)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1, 0, 0 ]

  def test_rational_id_and_function( self ) -> None:
    '''Rational applied to id and cube functions.'''
    Data = torch.tensor( [ [ 1., -2. ] ] )
    RegNames = [ "p" ]
    y = torch.tensor( [ 3. ] )
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( y, Data, RegNames, Functions, [ True, False, True ] )

    rat_id = -y.view( -1, 1 ) * Data
    rat_cube = -y.view( -1, 1 ) * ( Data**3 )
    expected = torch.cat( [ Data, Data**2, Data**3, rat_id, rat_cube ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )

    expected_names = np.array( [ "p", "sq(p)", "cube(p)", "~/(p)", "~/cube(p)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 1, 2, 0, 0 ]

  def test_only_id_with_rational( self ) -> None:
    '''Only id function with rational enabled.'''
    Data = torch.tensor( [ [ 2. ] ] )
    y = torch.tensor( [ 5. ] )
    out_data, out_names, M = NonLinearizer( y, Data, [ "z" ], [ id_func ], [ True ] )
    expected = torch.cat( [ Data, -y.view( -1, 1 ) * Data ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    assert np.array_equal( out_names, np.array( [ "z", "~/(z)" ] ) )
    assert M == [ 0, 0 ]

  # ============ Edge cases with y =======================================

  def test_y_none_and_rational_needed( self ) -> None:
    '''None y with rational raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    with pytest.raises( AssertionError, match = "y must be passed if MakeRational is not None" ):
      NonLinearizer( None, Data, [ "a", "b" ], [ id_func ], [ True ] )

  def test_y_length_mismatch( self ) -> None:
    '''y length mismatch raises AssertionError.'''
    Data = torch.rand( 4, 2 )
    y = torch.rand( 3 )
    with pytest.raises( AssertionError, match = "y's length does not match" ):
      NonLinearizer( y, Data, [ "a", "b" ], [ id_func ], None )

  def test_y_scalar_raises( self ) -> None:
    '''Scalar y raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    y = torch.tensor( 5.0 )
    with pytest.raises( AssertionError, match = "'y' must not be a scalar" ):
      NonLinearizer( y, Data, [ "a", "b" ], [ id_func ], None )

  def test_y_3d_raises( self ) -> None:
    '''3D y raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    y = torch.rand( 3, 1, 1 )
    with pytest.raises( AssertionError, match = "'y' must be 1D or a 2D tensor" ):
      NonLinearizer( y, Data, [ "a", "b" ], [ id_func ], None )

  def test_y_matrix_raises( self ) -> None:
    '''Full 2D matrix y raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    y = torch.rand( 3, 4 ) # full 2D matrix
    with pytest.raises( AssertionError, match = "must be a column vector" ):
      NonLinearizer( y, Data, [ "a", "b" ], [ id_func ], None )

  def test_y_row_vector( self ) -> None:
    '''Row vector y is accepted.'''
    Data = torch.rand( 3, 2 )
    y = torch.rand( 1, 3 ) # row vector (1,3)
    out, out_names, M = NonLinearizer( y, Data, [ "a", "b" ], [ id_func, sq_func ], None )
    assert out.shape == ( 3, 4 ) # 2 cols id + 2 cols sq
    expected_data = torch.cat( [ Data, Data**2 ], dim = 1 )
    self._assert_tensors_equal( out, expected_data )
    expected_names = np.array( [ "a", "b", "sq(a)", "sq(b)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1 ]

  def test_y_column_vector( self ) -> None:
    '''Column vector y is accepted.'''
    Data = torch.rand( 3, 2 )
    y = torch.rand( 3, 1 ) # column vector
    out, out_names, M = NonLinearizer( y, Data, [ "a", "b" ], [ id_func, sq_func ], None )
    assert out.shape == ( 3, 4 )
    expected_data = torch.cat( [ Data, Data**2 ], dim = 1 )
    self._assert_tensors_equal( out, expected_data )
    expected_names = np.array( [ "a", "b", "sq(a)", "sq(b)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1 ]

  # ============ Input validations =======================================

  def test_data_not_tensor_raises( self ) -> None:
    '''Non-tensor data raises AssertionError.'''
    with pytest.raises( AssertionError, match = "Input data must be a torch.Tensor" ):
      NonLinearizer( None, [ [ 1, 2 ] ], [ "a" ], [ id_func ] )

  def test_data_1d_raises( self ) -> None:
    '''1D data raises AssertionError.'''
    Data = torch.rand( 5 )
    with pytest.raises( AssertionError, match = "must be a 2D" ):
      NonLinearizer( None, Data, [ "x" ], [ id_func ] )

  def test_functions_not_list_raises( self ) -> None:
    '''Non-list Functions raises AssertionError.'''
    Data = torch.rand( 2, 2 )
    with pytest.raises( AssertionError, match = "'Functions'argument name must be a list" ):
      NonLinearizer( None, Data, [ "a", "b" ], ( id_func, ), None )

  def test_functions_empty_raises( self ) -> None:
    '''Empty Functions list raises AssertionError.'''
    Data = torch.rand( 2, 2 )
    with pytest.raises( AssertionError, match = "non-empty list" ):
      NonLinearizer( None, Data, [ "a", "b" ], [] )

  def test_rational_id_and_function( self ) -> None:
    '''Rational combined with id and cube.'''
    Data = torch.tensor( [ [ 1. ], [ -2. ] ] ) # shape (2,1)
    RegNames = [ "p" ]
    y = torch.tensor( [ 3., 4. ] )
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( y, Data, RegNames, Functions, [ True, False, True ] )
    rat_id = -y.view( -1, 1 ) * Data
    rat_cube = -y.view( -1, 1 ) * ( Data**3 )
    expected = torch.cat( [ Data, Data**2, Data**3, rat_id, rat_cube ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    expected_names = np.array( [ "p", "sq(p)", "cube(p)", "~/(p)", "~/cube(p)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 1, 2, 0, 0 ]

  def test_makerational_length_mismatch( self ) -> None:
    '''MakeRational length mismatch raises AssertionError.'''
    Data = torch.rand( 2, 2 )
    y = torch.rand( 2 )
    with pytest.raises( AssertionError, match = "length of MakeRational doesn't match" ):
      NonLinearizer( y, Data, [ "a", "b" ], [ id_func, sq_func ], [ True ] )

  def test_makerational_empty_list_treated_as_none( self ) -> None:
    '''Empty list MakeRational treated as None.'''
    Data = torch.rand( 3, 1 )
    Functions = [ id_func, sq_func ]
    y = torch.rand( 3 )
    # Now [] should be converted to None internally -> no rational, no error
    out_data, out_names, M = NonLinearizer( y, Data, [ "x" ], Functions, [] )
    expected = torch.cat( [ Data, Data**2 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    assert list( out_names ) == [ "x", "sq(x)" ]
    assert M == [ 0, 1 ]

  def test_makerational_empty_tuple_treated_as_none( self ) -> None:
    '''Empty tuple MakeRational treated as None.'''
    Data = torch.rand( 3, 1 )
    Functions = [ id_func, sq_func ]
    y = torch.rand( 3 )
    out_data, out_names, M = NonLinearizer( y, Data, [ "x" ], Functions, () )
    expected = torch.cat( [ Data, Data**2 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    assert list( out_names ) == [ "x", "sq(x)" ]
    assert M == [ 0, 1 ]

  def test_only_id_function_warning( self, capsys ) -> None:
    '''Only id function prints warning.'''
    Data = torch.rand( 3, 2 )
    out_data, out_names, M = NonLinearizer( None, Data, [ "a", "b" ], [ id_func ], None )
    captured = capsys.readouterr().out
    assert "WARNING: No transformations (Functions) or MakeRational instructions" in captured
    self._assert_tensors_equal( out_data, Data )
    assert np.array_equal( out_names, np.array( [ "a", "b" ] ) )
    assert M == [ 0, 0 ]

  # ============ Zero columns edge case ==================================

  def test_zero_columns_data( self ) -> None:
    '''Zero-column data works correctly.'''
    N = 4
    Data = torch.rand( N, 0 )
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( None, Data, [], Functions, None )
    assert out_data.shape == ( N, 0 )
    assert len( out_names ) == 0
    assert M == []

    # with rational
    y = torch.rand( N )
    out2, out_names2, M2 = NonLinearizer( y, Data, [], Functions, [ True, False, True ] )
    assert out2.shape == ( N, 0 )
    assert len( out_names2 ) == 0
    assert M2 == []

  # ============ Exact computational tests ===============================

  def test_exact_values_id_sq_cube( self ) -> None:
    '''Exact values for id, sq, cube with rational.'''
    Data = torch.tensor( [ [ 1., 2., 3. ], [ 4., 5., 6. ] ] )
    RegNames = [ "x0", "x1", "x2" ]
    y = torch.tensor( [ -2., 0.5 ] )
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( y, Data, RegNames, Functions, [ False, False, True ] )

    # Manually compute: columns: id(3), sq(3), cube(3), rational cube(3)
    id_part = Data
    sq_part = Data**2
    cube_part = Data**3
    rat_cube = -y.view( -1, 1 ) * cube_part

    expected_data = torch.cat( [ id_part, sq_part, cube_part, rat_cube ], dim = 1 )
    self._assert_tensors_equal( out_data, expected_data )

    # names
    id_names = [ "x0", "x1", "x2" ]
    sq_names = [ "sq(x0)", "sq(x1)", "sq(x2)" ]
    cube_names = [ "cube(x0)", "cube(x1)", "cube(x2)" ]
    rat_names = [ "~/cube(x0)", "~/cube(x1)", "~/cube(x2)" ]
    expected_names = np.array( id_names + sq_names + cube_names + rat_names )
    assert np.array_equal( out_names, expected_names )

    # M: id (0*3), sq (1*3), cube (2*3), rational cube (0*3)
    assert M == [ 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0 ]

  def test_rational_uses_original_names( self ) -> None:
    '''Rational column names use original function names.'''
    Data = torch.ones( 2, 1 )
    RegNames = [ "temp" ]
    Functions = [ id_func, NonLinearity( "double", lambda x : 2 * x ) ]
    y = torch.tensor( [ 1., 2. ] )
    _, out_names, _ = NonLinearizer( y, Data, RegNames, Functions, [ True, True ] )
    expected = np.array( [ "temp", "double(temp)", "~/(temp)", "~/double(temp)" ] )
    assert np.array_equal( out_names, expected )

  def test_return_types( self ) -> None:
    '''Return types, dtypes, and M values are correct.'''
    Data = torch.rand( 2, 1 )
    out_data, out_names, M = NonLinearizer( None, Data, [ "x" ], [ id_func, sq_func ], None )
    assert isinstance( out_data, torch.Tensor )
    assert out_data.dtype == Data.dtype
    assert isinstance( out_names, np.ndarray )
    assert out_names.dtype.kind in ( 'U', 'S' )
    assert isinstance( M, list )
    assert all( isinstance( m, int ) for m in M )
    assert M == [ 0, 1 ]

  # ============ Missing validation error cases ===========================

  def test_first_function_not_id_raises( self ) -> None:
    '''First function not "id" raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    with pytest.raises( AssertionError, match = "0th function in the 'Functions' list must be 'id'" ):
      NonLinearizer( None, Data, [ "a", "b" ], [ sq_func ], None )

  def test_y_not_tensor_raises( self ) -> None:
    '''Non-tensor y raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    with pytest.raises( AssertionError, match = "'y' argument must be a torch.Tensor" ):
      NonLinearizer( [ 1, 2, 3 ], Data, [ "a", "b" ], [ id_func ], None )

  def test_regnames_2d_raises( self ) -> None:
    '''2D RegNames raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    RegNames = np.array( [ [ "a", "b" ], [ "c", "d" ] ] )
    with pytest.raises( AssertionError, match = "RegNames argument must be a 1D array" ):
      NonLinearizer( None, Data, RegNames, [ id_func ], None )

  def test_regnames_length_mismatch_raises( self ) -> None:
    '''RegNames length mismatch with Data columns raises AssertionError.'''
    Data = torch.rand( 3, 2 )
    with pytest.raises( AssertionError, match = "RegNames argument must have the same length" ):
      NonLinearizer( None, Data, [ "a" ], [ id_func ], None )

  # ============ Additional edge cases ====================================

  def test_negative_data_values( self ) -> None:
    '''NonLinearizer handles negative data values correctly.'''
    Data = torch.tensor( [ [ -1.0, 2.0 ], [ 3.0, -4.0 ] ] )
    RegNames = [ "x", "y" ]
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( None, Data, RegNames, Functions, None )
    expected = torch.cat( [ Data, Data**2, Data**3 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    expected_names = np.array( [ "x", "y", "sq(x)", "sq(y)", "cube(x)", "cube(y)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1, 2, 2 ]

  def test_zero_data( self ) -> None:
    '''NonLinearizer handles all-zero data.'''
    Data = torch.zeros( 3, 2 )
    RegNames = [ "a", "b" ]
    Functions = [ id_func, sq_func, cube_func ]
    out_data, out_names, M = NonLinearizer( None, Data, RegNames, Functions, None )
    expected = torch.cat( [ Data, Data**2, Data**3 ], dim = 1 )
    self._assert_tensors_equal( out_data, expected )
    expected_names = np.array( [ "a", "b", "sq(a)", "sq(b)", "cube(a)", "cube(b)" ] )
    assert np.array_equal( out_names, expected_names )
    assert M == [ 0, 0, 1, 1, 2, 2 ]
