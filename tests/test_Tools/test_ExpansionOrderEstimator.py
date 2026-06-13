# test_expansion_order_estimator.py
import numpy as np
import torch as tor
import pytest
from typing import Any
from unittest.mock import patch, MagicMock, call

from NARMAX.Tools import ExpansionOrderEstimator


class TestExpansionOrderEstimator:
  @pytest.fixture
  def mock_dependencies( self ) -> tuple[ MagicMock, MagicMock, MagicMock, MagicMock ]:
    '''Fixture that patches CTors, ComputeERR, tqdm, and plt.'''
    with patch( 'NARMAX.Tools.CTors' ) as mock_ctors, patch( 'NARMAX.Tools.ComputeERR' ) as mock_compute_err, patch( 'NARMAX.Tools.tqdm.tqdm' ) as mock_tqdm, patch( 'NARMAX.Tools.plt' ) as mock_plt:
      yield mock_ctors, mock_compute_err, mock_tqdm, mock_plt

  def test_assertions( self, mock_dependencies ) -> None:
    '''Input validation assertions raise on invalid arguments.'''
    mock_ctors, _, _, _ = mock_dependencies
    x = tor.randn( 50 )
    y_diff_len = tor.randn( 51 )
    with pytest.raises( AssertionError, match = "same shape" ):
      ExpansionOrderEstimator( x, y_diff_len, Plot = False )

    x_2d = tor.randn( 50, 1 )
    y_2d = tor.randn( 50, 1 )
    with pytest.raises( AssertionError, match = "not a \\(p,\\)-shaped" ):
      ExpansionOrderEstimator( x_2d, y_2d, Plot = False )
    with pytest.raises( AssertionError, match = "not a \\(p,\\)-shaped" ):
      ExpansionOrderEstimator( y_2d, x_2d, Plot = False )

    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      ExpansionOrderEstimator( x = tor.randn( 10 ), y = tor.randn( 10 ),
                                    MaxOrder = 0, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      ExpansionOrderEstimator( tor.randn( 10 ), tor.randn( 10 ),
                                    MaxOrder = 2.5, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      ExpansionOrderEstimator( x = tor.randn( 10 ), y = tor.randn( 10 ),
                                    MaxOrder = -1, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      ExpansionOrderEstimator( tor.randn( 10 ), tor.randn( 10 ),
                                    MaxOrder = 3.0, Plot = False )

  def test_basic_break_when_threshold_met( self, mock_dependencies ) -> None:
    '''Loop breaks when variance threshold is reached.'''
    mock_ctors, mock_compute_err, mock_tqdm, mock_plt = mock_dependencies
    x = tor.randn( 30 )
    y = tor.randn( 30 )
    # Simulate Lagger returning some regressors
    y_cut = tor.randn( 30 ) + 5.0 # non‑zero mean to test centering
    reg_mat = tor.randn( 30, 3 )
    reg_names = [ 'a', 'b', 'c' ]
    mock_ctors.Lagger.return_value = ( y_cut, reg_mat, reg_names )

    # Expander: just return the same matrix (but shape may change)
    mock_ctors.Expander.return_value = ( reg_mat, reg_names )

    # ERR values: order 1 = 0.6, order 2 = 0.9, order 3 would be 0.99 but threshold 0.98 met at 2
    err_sums = [ 0.6, 0.95 ] # after order 1 and 2
    # ComputeERR returns array that sums to err_sum
    def compute_err_side( y_call, ds_call ):
      # Called once per order; we'll yield a new sum each time
      # We need to return an array whose sum equals the next err_sum
      nonlocal err_sums
      val = err_sums.pop( 0 )
      return np.array( [ val * 0.5, val * 0.5 ] ) # two elements sum to val
    mock_compute_err.side_effect = compute_err_side

    order, explained_var = ExpansionOrderEstimator(
            x, y, MaxOrder = 3, VarianceAcceptThreshold = 0.95, Plot = False
        )

    assert order == 2
    assert isinstance( order, int )
    assert isinstance( explained_var, np.ndarray )
    assert explained_var.dtype == np.float64
    assert len( explained_var ) == 3 # 0, order1, order2
    assert explained_var[ 0 ] == 0.0
    assert explained_var[ 1 ] == 0.6
    assert explained_var[ 2 ] == 0.95
    assert np.all( explained_var >= 0.0 )
    assert np.all( np.diff( explained_var ) >= -1e-15 )  # non-decreasing

    # Verify y_cut was centered (mean ~0) before passing to ComputeERR
    args_y = mock_compute_err.call_args_list[ 0 ][ 0 ][ 0 ]
    assert tor.abs( args_y.mean() ) < 1e-6

    # Verify regressor matrix was centered
    args_ds = mock_compute_err.call_args_list[ 0 ][ 0 ][ 1 ]
    assert tor.all( tor.abs( args_ds.mean( dim = 0 ) ) < 1e-6 )

    # Check tqdm was used and closed
    mock_tqdm.return_value.update.assert_called()
    mock_tqdm.return_value.close.assert_called()

  def test_threshold_not_met_returns_None( self, mock_dependencies ) -> None:
    '''None returned when threshold is never reached.'''
    mock_ctors, mock_compute_err, mock_tqdm, mock_plt = mock_dependencies
    x = tor.randn( 20 )
    y = tor.randn( 20 )
    mock_ctors.Lagger.return_value = ( tor.randn( 20 ), tor.randn( 20, 2 ), [] )
    mock_ctors.Expander.return_value = ( tor.randn( 20, 5 ), [] )

    # All orders below threshold 0.99
    mock_compute_err.return_value = np.array( [ 0.3, 0.3, 0.2 ] ) # sum 0.8

    order, explained_var = ExpansionOrderEstimator(
            x, y, MaxOrder = 3, VarianceAcceptThreshold = 0.99, Plot = False
        )
    assert order is None
    assert isinstance( explained_var, np.ndarray )
    assert explained_var.dtype == np.float64
    assert len( explained_var ) == 4 # 0 + orders 1..3
    assert explained_var[ 0 ] == 0.0
    assert explained_var[ -1 ] < 0.99
    assert np.all( explained_var >= 0.0 )

  def test_plot_true( self, mock_dependencies ) -> None:
    '''Plotting is triggered when Plot=True.'''
    mock_ctors, mock_compute_err, mock_tqdm, mock_plt = mock_dependencies
    x = tor.randn( 15 )
    y = tor.randn( 15 )
    mock_ctors.Lagger.return_value = ( tor.randn( 15 ), tor.randn( 15, 2 ), [] )
    mock_ctors.Expander.return_value = ( tor.randn( 15, 2 ), [] )
    # threshold met after order 1
    mock_compute_err.side_effect = [ np.array( [ 0.99 ] ) ]

    mock_plt.subplots.return_value = ( MagicMock(), MagicMock() )

    order, explained_var = ExpansionOrderEstimator(
            x, y, MaxOrder = 2, VarianceAcceptThreshold = 0.95, Plot = True
        )

    # Check return types
    assert isinstance( order, int )
    assert isinstance( explained_var, np.ndarray )
    assert explained_var.dtype == np.float64

    # Check plot calls
    mock_plt.subplots.assert_called_once()
    fig, ax = mock_plt.subplots.return_value
    ax.plot.assert_called_once()
    # The plotted data should be 100 * explained_var
    plotted_data = ax.plot.call_args[ 0 ][ 0 ]
    np.testing.assert_array_almost_equal( plotted_data, 100 * explained_var )

    # Horizontal threshold line
    ax.axhline.assert_called_once_with( y = 100 * 0.95, c = 'purple', linewidth = 1.5, linestyle = '--' )
    ax.grid.assert_called()
    ax.legend.assert_called()
    ax.set.assert_called()
    fig.tight_layout.assert_called_once()

    # Check xticks match explained_var length
    xticks_args = ax.set_xticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( xticks_args, np.arange( len( explained_var ) ) )
    xticklabels_args = ax.set_xticklabels.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( xticklabels_args, np.arange( len( explained_var ) ) )

  def test_plot_false_no_plot_calls( self, mock_dependencies ) -> None:
    '''No plot calls when Plot=False.'''
    mock_ctors, mock_compute_err, mock_tqdm, mock_plt = mock_dependencies
    x = tor.randn( 10 )
    y = tor.randn( 10 )
    mock_ctors.Lagger.return_value = ( tor.randn( 10 ), tor.randn( 10, 1 ), [] )
    mock_ctors.Expander.return_value = ( tor.randn( 10, 1 ), [] )
    mock_compute_err.return_value = np.array( [ 0.5 ] )
    order, explained_var = ExpansionOrderEstimator( x, y, MaxOrder = 1, Plot = False )
    mock_plt.subplots.assert_not_called()
    # With default threshold 0.98 and ERR sum 0.5, the threshold is not met
    assert order is None
    assert isinstance( explained_var, np.ndarray )
    assert explained_var.dtype == np.float64
    assert explained_var[ 0 ] == 0.0
    assert len( explained_var ) == 2

  def test_maxorder_1( self, mock_dependencies ) -> None:
    '''MaxOrder=1 with threshold met returns order=1.'''
    mock_ctors, mock_compute_err, _, _ = mock_dependencies
    x = tor.randn( 25 )
    y = tor.randn( 25 )
    mock_ctors.Lagger.return_value = ( tor.randn( 25 ), tor.randn( 25, 1 ), [] )
    mock_ctors.Expander.return_value = ( tor.randn( 25, 1 ), [] )
    mock_compute_err.return_value = np.array( [ 0.7 ] )
    order, ev = ExpansionOrderEstimator( x, y, MaxOrder = 1, VarianceAcceptThreshold = 0.7, Plot = False )
    assert order == 1
    assert isinstance( order, int )
    assert isinstance( ev, np.ndarray )
    assert ev.dtype == np.float64
    assert ev.tolist() == [ 0.0, 0.7 ]
    assert np.all( np.diff( ev ) >= -1e-15 )  # non-decreasing

  def test_call_sequence_lagger_expander( self, mock_dependencies ) -> None:
    '''Verifies Lagger/Expander call sequence with MaxLags.'''
    mock_ctors, mock_compute_err, _, _ = mock_dependencies
    x = tor.randn( 30 )
    y = tor.randn( 30 )
    y_ret = tor.randn( 30 ) + 3.0
    reg_ret = tor.randn( 30, 4 )
    names_ret = [ 'x1', 'y1' ]
    mock_ctors.Lagger.return_value = ( y_ret, reg_ret, names_ret )
    mock_ctors.Expander.return_value = ( reg_ret, names_ret )
    mock_compute_err.return_value = np.array( [ 0.99 ] )

    order, explained_var = ExpansionOrderEstimator( x, y, MaxOrder = 2, MaxLags = ( 5, 5 ),
                                VarianceAcceptThreshold = 0.98, Plot = False )
    # Threshold met after order 1 (ERR sum 0.99 > 0.98)
    assert order == 1
    assert isinstance( order, int )
    assert isinstance( explained_var, np.ndarray )
    assert explained_var.dtype == np.float64
    assert len( explained_var ) == 2  # 0 + order 1
    assert explained_var[ 0 ] == 0.0
    # Lagger called with (x,y) and MaxLags
    mock_ctors.Lagger.assert_called_once()
    lagger_args, _ = mock_ctors.Lagger.call_args
    assert len( lagger_args ) == 2
    assert isinstance( lagger_args[ 0 ], tuple ) and len( lagger_args[ 0 ] ) == 2
    assert tor.equal( lagger_args[ 0 ][ 0 ], x )
    assert tor.equal( lagger_args[ 0 ][ 1 ], y )
    assert lagger_args[ 1 ] == ( 5, 5 )
    # Expander called with the regressors and ExpansionOrder=1 then maybe 2
    assert mock_ctors.Expander.call_count == 1 # because threshold met after order 1
    mock_ctors.Expander.assert_called_with( reg_ret, names_ret, ExpansionOrder = 1 )
