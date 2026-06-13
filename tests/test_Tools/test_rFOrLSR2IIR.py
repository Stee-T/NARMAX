import pytest
import numpy as np
from NARMAX.Tools import rFOrLSR2IIR # adjust import

class TestrFOrLSR2IIR:
  '''Unit tests for rFOrLSR2IIR - FOrLSR to IIR coefficient conversion.'''

  # ---------------------------------------------------------------
  # Valid conversions
  # ---------------------------------------------------------------
  def test_basic_mixed_terms( self ) -> None:
    '''Mixed x and y terms produce correct b and a coefficients.'''
    theta = [ 0.5, -0.3, 0.8, -0.2 ]
    L = [ 0, 2, 3, 5 ]
    reg_names = [ 'x[k]', 'x[k-1]', 'y[k-1]', 'x[k-2]', 'y[k-2]', 'y[k-3]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert a[ 0 ] == 1.0
    expected_b = np.array( [ 0.5, 0.0, 0.8, 0.0 ] ) # x[0]=0.5, x[2]=0.8, others 0
    expected_a = np.array( [ 1.0, 0.3, 0.0, 0.2 ] ) # y[1]=-(-0.3)=0.3, y[3]=-(-0.2)=0.2
    np.testing.assert_array_almost_equal( b, expected_b )
    np.testing.assert_array_almost_equal( a, expected_a )

  def test_only_x_terms( self ) -> None:
    '''Only x terms produce b coefficients and a=[1,0,...].'''
    theta = [ 1.0, 2.0 ]
    L = [ 0, 1 ]
    reg_names = [ 'x[k]', 'x[k-1]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert a[ 0 ] == 1.0
    np.testing.assert_array_almost_equal( b, np.array( [ 1.0, 2.0 ] ) )
    np.testing.assert_array_almost_equal( a, np.array( [ 1.0, 0.0 ] ) )

  def test_only_y_terms_sign_flip( self ) -> None:
    '''Only y terms have sign flipped for denominator coefficients.'''
    theta = [ 0.9, -0.5 ]
    L = [ 0, 1 ]
    reg_names = [ 'y[k-1]', 'y[k-2]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert a[ 0 ] == 1.0
    # a[1] = -0.9, a[2] = -(-0.5)=0.5
    expected_a = np.array( [ 1.0, -0.9, 0.5 ] )
    np.testing.assert_array_almost_equal( a, expected_a )
    np.testing.assert_array_almost_equal( b, np.zeros( 3 ) ) # no x terms

  def test_max_lag_determines_vector_lengths( self ) -> None:
    '''Max lag determines the length of b and a vectors.'''
    theta = [ 0.1 ]
    L = [ 0 ]
    reg_names = [ 'x[k-5]', 'y[k-5]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    # Max lag = 5 -> vectors of length 6 (0..5)
    assert len( b ) == 6
    assert len( a ) == 6
    assert a[ 0 ] == 1.0
    assert b[ 5 ] == 0.1
    np.testing.assert_array_almost_equal( a[ 1 : ], np.zeros( 5 ) )

  # ---------------------------------------------------------------
  # Input validation & errors
  # ---------------------------------------------------------------
  def test_empty_L_raises( self ) -> None:
    '''Empty L list raises ValueError.'''
    with pytest.raises( ValueError, match = "L must not be empty" ):
      rFOrLSR2IIR( [], [], [ 'x[k]' ] )

  def test_length_mismatch_raises( self ) -> None:
    '''Length mismatch between theta and L raises ValueError.'''
    with pytest.raises( ValueError, match = "Length mismatch: L has 1 elements, theta has 2" ):
      rFOrLSR2IIR( [ 1, 2 ], [ 0 ], [ 'x[k]' ] )

  def test_index_out_of_bounds_raises( self ) -> None:
    '''L index out of bounds raises ValueError.'''
    with pytest.raises( ValueError, match = "Regressor index 1 out of bounds" ):
      rFOrLSR2IIR( [ 1.0 ], [ 1 ], [ 'x[k]' ] ) # only index 0 exists

  def test_duplicate_indices_raises( self ) -> None:
    '''Duplicate regressor indices raise ValueError.'''
    with pytest.raises( ValueError, match = "Duplicate regressor indices in L are not allowed" ):
      rFOrLSR2IIR( [ 0.1, 0.2 ], [ 0, 0 ], [ 'x[k]', 'x[k-1]' ] )

  def test_yk_zero_delay_raises( self ) -> None:
    '''y[k] term is not allowed.'''
    with pytest.raises( ValueError, match = r"y\[k\] term is not allowed; denominator a\[0\]" ):
      rFOrLSR2IIR( [ 0.5 ], [ 0 ], [ 'y[k]' ] )

  def test_invalid_term_format_raises( self ) -> None:
    '''Invalid term format raises ValueError.'''
    reg_names = [ 'z[k-1]' ] # z not allowed
    with pytest.raises( ValueError, match = r"Invalid term: z\[k-1\]" ):
      rFOrLSR2IIR( [ 1.0 ], [ 0 ], reg_names )

  def test_duplicate_variable_delay_after_parsing_raises( self ) -> None:
    '''Duplicate variable+delay mapping raises ValueError.'''
    # Two different regressor indices map to same variable & delay
    reg_names = [ 'x[k]', 'x[k]' ]
    with pytest.raises( ValueError, match = "Duplicate regressor mapping found" ):
      rFOrLSR2IIR( [ 0.5, 0.6 ], [ 0, 1 ], reg_names )

  def test_duplicate_delay_across_x_and_y_is_allowed( self ) -> None:
    '''Same delay for x and y is allowed (different variables).'''
    # x[k-1] and y[k-1] are different variables -> allowed
    theta = [ 0.1, 0.2 ]
    L = [ 0, 1 ]
    reg_names = [ 'x[k-1]', 'y[k-1]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert len( b ) == 2
    assert len( a ) == 2
    assert a[ 0 ] == 1.0
    assert b[ 0 ] == 0.0
    assert b[ 1 ] == 0.1
    assert a[ 1 ] == -0.2 # y-term sign flipped

  # ---------------------------------------------------------------
  # Edge cases
  # ---------------------------------------------------------------
  def test_empty_regressor_list_but_zero_lag_only( self ) -> None:
    '''All x[k] terms work correctly.'''
    # All x[k] (zero lag) is fine
    theta = [ 3.0 ]
    L = [ 0 ]
    reg_names = [ 'x[k]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert len( b ) == 1
    assert len( a ) == 1
    np.testing.assert_array_almost_equal( b, np.array( [ 3.0 ] ) )
    np.testing.assert_array_almost_equal( a, np.array( [ 1.0 ] ) )

  def test_large_delays_and_coefficients( self ) -> None:
    '''Large delays produce correctly-sized vectors with coefficients at the right positions.'''
    theta = [ 1e6, -1e6 ]
    L = [ 0, 1 ]
    reg_names = [ 'x[k-100]', 'y[k-100]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert b[ 100 ] == 1e6
    assert a[ 100 ] == 1e6 # -(-1e6)
    assert len( b ) == 101
    assert len( a ) == 101
    assert a[ 0 ] == 1.0
    assert b[ 0 ] == 0.0
    # All coefficients before index 100 should be zero
    np.testing.assert_array_almost_equal( b[ :100 ], np.zeros( 100 ) )
    np.testing.assert_array_almost_equal( a[ 1:100 ], np.zeros( 99 ) )

  # ---------------------------------------------------------------
  # Additional edge cases
  # ---------------------------------------------------------------
  def test_non_contiguous_delays_zeros_filled( self ) -> None:
    '''Gaps in delays produce zeros in the coefficient vectors.'''
    theta = [ 0.5, -0.3, 0.2 ]
    L = [ 0, 1, 2 ]
    reg_names = [ 'x[k]', 'y[k-3]', 'x[k-5]' ]
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    # max lag = 5, so vectors are length 6
    expected_b = np.array( [ 0.5, 0.0, 0.0, 0.0, 0.0, 0.2 ] )
    expected_a = np.array( [ 1.0, 0.0, 0.0, 0.3, 0.0, 0.0 ] ) # a[3] = -(-0.3)
    np.testing.assert_array_almost_equal( b, expected_b )
    np.testing.assert_array_almost_equal( a, expected_a )
    assert len( b ) == 6
    assert len( a ) == 6

  def test_input_as_numpy_arrays( self ) -> None:
    '''Function accepts numpy arrays as input.'''
    theta = np.array( [ 0.5, -0.3 ] )
    L = np.array( [ 0, 1 ] )
    reg_names = np.array( [ 'x[k]', 'y[k-1]' ] )
    b, a = rFOrLSR2IIR( theta, L, reg_names )
    assert isinstance( b, np.ndarray )
    assert isinstance( a, np.ndarray )
    assert np.issubdtype( b.dtype, np.floating )
    assert np.issubdtype( a.dtype, np.floating )
    assert a[ 0 ] == 1.0
    np.testing.assert_array_almost_equal( b, np.array( [ 0.5, 0.0 ] ) )
    np.testing.assert_array_almost_equal( a, np.array( [ 1.0, 0.3 ] ) )
