# test_compute_err.py
import numpy as np
import torch as tor
import pytest
from NARMAX.Tools import ComputeERR


class TestComputeERR:
  def test_empty_ds( self ) -> None:
    '''When Ds has no columns, return an empty array.'''
    y = tor.randn( 10 )
    Ds = tor.empty( ( 10, 0 ) )
    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.shape == ( 0, )
    assert err.dtype == np.float64
    assert np.all( err == 0.0 )
    assert err.sum() == 0.0
    assert not np.any( np.isnan( err ) )

  def test_zero_variance_raises( self ) -> None:
    '''y with variance ≤ 1e-15 should raise ValueError.'''
    y = tor.zeros( 10 ) # constant → variance 0
    Ds = tor.randn( 10, 1 )
    with pytest.raises( ValueError, match = "zero variance" ):
      ComputeERR( y, Ds )

    # y with variance just above threshold should work
    y_ok = tor.randn( 10 ) * 1e-5 # variance ~ 1e-10
    err = ComputeERR( y_ok, tor.randn( 10, 1 ) ) # no exception
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 1, )
    assert 0.0 <= err[ 0 ] <= 1.0 + 1e-12
    assert not np.isnan( err[ 0 ] )

  def test_single_column( self ) -> None:
    '''One regressor: ERR = (dot(y, Omega)^2) / (||Omega||^2 * var(y)).'''
    y = tor.tensor( [ 1.0, -2.0, 3.0, -1.0 ] ) # zero‑mean
    Omega = tor.tensor( [ 2.0, -1.0, 0.0, 4.0 ] )
    Ds = Omega[ :, None ] # shape (4,1)
    s2y = ( y @ y ).item()
    n_Omega_real = tor.sum( Omega**2 ).item()
    expected_err = ( Omega @ y ).item()**2 / ( n_Omega_real * s2y )

    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 1, )
    assert np.isclose( err[ 0 ], expected_err )
    assert 0.0 <= err[ 0 ] <= 1.0 + 1e-12
    assert not np.isnan( err[ 0 ] )

  def test_multiple_columns( self ) -> None:
    '''Verify ERR values for 3 columns using manual Gram–Schmidt.'''
    tor.manual_seed( 42 )
    y = tor.randn( 20 ) - tor.randn( 20 ).mean() # zero‑mean
    Ds = tor.randn( 20, 3 )

    # Reference calculation (same algorithm as the implementation)
    s2y = ( y @ y ).item()
    err = ComputeERR( y, Ds )

    # Recompute manually
    Psi = tor.empty( ( 20, 0 ) )
    Psi_n = tor.empty( ( 20, 0 ) )
    manual_err = np.full( 3, 0.0 )
    for col in range( 3 ):
      if ( col == 0 ): Omega = Ds[ :, 0 ]
      else: Omega = Ds[ :, col ] - Psi_n @ ( Psi.T @ Ds[ :, col ] )
      n_Omega = max( tor.sum( Omega**2 ).item(), 1e-12 )
      manual_err[ col ] = ( Omega @ y ).item()**2 / ( n_Omega * s2y )
      Psi = tor.column_stack( ( Psi, Omega ) )
      Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) )

    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert np.allclose( err, manual_err )
    assert np.all( err >= 0.0 )
    assert np.all( err <= 1.0 + 1e-12 )
    assert np.sum( err ) <= 1.0 + 1e-12
    assert not np.any( np.isnan( err ) )

  def test_early_exit( self ) -> None:
    '''If cumulative ERR reaches 1.0, remaining columns keep their initial 0.0.'''
    y = tor.randn( 10 ) - tor.randn( 10 ).mean()
    # First column perfectly correlated with y (explains 100%)
    col1 = y * 2.0
    col2 = tor.randn( 10 ) # not used
    Ds = tor.column_stack( ( col1, col2 ) )

    err = ComputeERR( y, Ds )

    # First ERR ~1.0 (float rounding might be 0.999...), second stays 0.0
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 2, )
    assert err[ 0 ] >= 1.0 - 1e-12
    assert abs( err[ 1 ] ) < 1e-20 # never touched after early exit
    assert np.sum( err ) >= 1.0 - 1e-12

  def test_zero_column_fudge_factor( self ) -> None:
    '''All‑zero column must not cause division by zero; ERR=0.'''
    y = tor.randn( 10 ) - tor.randn( 10 ).mean()
    Ds = tor.column_stack( ( tor.zeros( 10 ), tor.randn( 10 ) ) )
    err = ComputeERR( y, Ds )
    # Zero column gives ERR=0, second column whatever
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 2, )
    assert err[ 0 ] == 0.0
    assert 0.0 <= err[ 1 ] <= 1.0 + 1e-12
    assert not np.any( np.isnan( err ) )

  def test_different_dtypes( self ) -> None:
    '''Function should work with float32 inputs, but output ERR is float64.'''
    y = tor.randn( 15, dtype = tor.float32 )
    Ds = tor.randn( 15, 2, dtype = tor.float32 )
    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 2, )
    assert np.all( err >= 0.0 )
    assert np.all( err <= 1.0 + 1e-12 )
    assert not np.any( np.isnan( err ) )

  def test_large_p( self ) -> None:
    '''Stress test with a large number of observations.'''
    p = 5000
    y = tor.randn( p ) - tor.randn( p ).mean()
    Ds = tor.randn( p, 5 )
    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 5, )
    assert np.all( err >= 0.0 )
    assert np.all( err <= 1.0 + 1e-12 )
    assert np.all( np.isfinite( err ) )

  def test_all_zero_regressors( self ) -> None:
    '''When all regressor columns are zero, the inline norm clips to epsilon
    and numerator (Omega @ y) is zero, so all ERR values are 0.0.'''
    y = tor.randn( 10 ) - tor.randn( 10 ).mean()
    Ds = tor.zeros( 10, 3 )
    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 3, )
    assert np.all( err == 0.0 )
    assert not np.any( np.isnan( err ) )

  def test_non_contiguous_input( self ) -> None:
    '''Sliced / non‑contiguous tensors must not raise and must produce valid ERR.'''
    y = tor.randn( 20 ) - tor.randn( 20 ).mean()
    Ds_full = tor.randn( 20, 4 )
    Ds = Ds_full[ :, ::2 ]  # non‑contiguous slice
    err = ComputeERR( y, Ds )
    assert isinstance( err, np.ndarray )
    assert err.dtype == np.float64
    assert err.shape == ( 2, )
    assert np.all( err >= 0.0 )
    assert np.all( err <= 1.0 + 1e-12 )
    assert not np.any( np.isnan( err ) )

  def test_1d_ds_raises_error( self ) -> None:
    '''When Ds is a 1‑D tensor, Ds.shape[ 1 ] raises IndexError.'''
    y = tor.randn( 10 )
    Ds = tor.randn( 10 )  # 1D, not 2D
    with pytest.raises( IndexError ):
      ComputeERR( y, Ds )
