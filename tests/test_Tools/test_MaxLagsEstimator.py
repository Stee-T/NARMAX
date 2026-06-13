# test_max_lags_estimator.py
import numpy as np
import torch as tor
import pytest
from unittest.mock import patch, MagicMock, call

from NARMAX.Tools import MaxLagsEstimator


class TestMaxLagsEstimator:
  '''Tests for MaxLagsEstimator grid search.'''
  @pytest.fixture
  def mock_all( self ) -> tuple[ MagicMock, MagicMock, MagicMock, MagicMock ]:
    with patch( 'NARMAX.Tools.CTors' ) as mock_ctors, patch( 'NARMAX.Tools.ComputeERR' ) as mock_compute_err, patch( 'NARMAX.Tools.tqdm.tqdm' ) as mock_tqdm, patch( 'NARMAX.Tools.plt' ) as mock_plt:
      yield mock_ctors, mock_compute_err, mock_tqdm, mock_plt

  def _configure_grid( self, mock_ctors: MagicMock, mock_compute_err: MagicMock, grid_vals: dict[ tuple[ int, int ], float ], max_lags: tuple[ int, int ] ) -> None:
    '''
    Helper that sets up Lagger/Expander/ComputeERR so that each (na,nb) cell
    receives the ERR sum from the dict `grid_vals` keyed by (na, nb).
    Cells not computed due to early optimisation will not be called.
    '''
    recorded_lags = None
    def lagger_side( Data, Lags ):
      nonlocal recorded_lags
      recorded_lags = Lags # (nb, na)
      # Return dummy tensors of appropriate size (p=10)
      p = 10
      return tor.randn( p ), tor.randn( p, 3 ), []
    mock_ctors.Lagger.side_effect = lagger_side
    mock_ctors.Expander.return_value = ( tor.randn( 10, 3 ), [] )

    # ComputeERR side effect uses recorded Lags to pick the right ERR sum
    def compute_err_side( y, ds ):
      nb, na = recorded_lags
      val = grid_vals[ ( na, nb ) ]
      return np.array( [ val ] ) # single‑element array, sum = val
    mock_compute_err.side_effect = compute_err_side

  def test_assertions( self, mock_all ) -> None:
    '''Input validation assertions raise on invalid arguments.'''
    mock_ctors, _, _, _ = mock_all
    x = tor.randn( 50 )
    y_diff = tor.randn( 51 )
    y_shorter = tor.randn( 49 )
    with pytest.raises( AssertionError, match = "same shape" ):
      MaxLagsEstimator( x, y_diff, ModelOrder = 2, Plot = False )
    with pytest.raises( AssertionError, match = "same shape" ):
      MaxLagsEstimator( x, y_shorter, ModelOrder = 2, Plot = False )
    with pytest.raises( AssertionError, match = "not a \\(p,\\)-shaped" ):
      MaxLagsEstimator( tor.randn( 50, 1 ), tor.randn( 50, 1 ), ModelOrder = 2, Plot = False )
    with pytest.raises( AssertionError, match = "not a \\(p,\\)-shaped" ):
      MaxLagsEstimator( tor.randn( 10, 5, 2 ), tor.randn( 10, 5, 2 ), ModelOrder = 2, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      MaxLagsEstimator( x, x, ModelOrder = 0, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      MaxLagsEstimator( x, x, ModelOrder = 1.5, Plot = False )
    with pytest.raises( AssertionError, match = "MaxOrder must be an int >= 1" ):
      MaxLagsEstimator( x, x, ModelOrder = -1, Plot = False )
    # No computation mocks should be called when assertions fire
    mock_ctors.Lagger.assert_not_called()

  def test_savefig_without_plot_raises( self, mock_all ) -> None:
    '''SaveFig without Plot=True raises AssertionError.'''
    mock_ctors, _, _, _ = mock_all
    x = tor.randn( 10 )
    with pytest.raises( AssertionError, match = "SaveFig can only be used if Plot = True" ):
      MaxLagsEstimator( x, x, ModelOrder = 1, Plot = False, SaveFig = "file.png" )

  def test_recommendations_min_xy( self, mock_all ) -> None:
    '''Min_XY, Min_Y, Min_X recommendations are computed correctly.'''
    mock_ctors, mock_compute_err, _, _ = mock_all
    # Grid of size (3,3) for MaxLags=(2,2) -> rows 0..2, cols 0..2
    max_lags = ( 2, 2 )
    threshold = 0.8
    # Define ERR sums for each cell (na, nb)
    grid_vals = {
            ( 0, 0 ): 0.5,
            ( 0, 1 ): 0.9, # na=0,nb=1 sum=1
            ( 0, 2 ): 0.9, # sum=2
            ( 1, 0 ): 0.9, # sum=1
            ( 1, 1 ): 0.85, # sum=2
            ( 1, 2 ): 0.95, # sum=3
            ( 2, 0 ): 0.95, # sum=2
            ( 2, 1 ): 0.95, # sum=3
            ( 2, 2 ): 0.95, # sum=4
        }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )

    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 2,
            MaxLags = max_lags, VarianceAcceptThreshold = threshold, Plot = False
        )

    # Min_XY: smallest na+nb. Candidates with sum=1: (0,1) and (1,0). argmin of sums gives first?
    # In code: idx = np.argwhere(valid); sums = idx[:,0]+idx[:,1]; best_idx = np.argmin(sums)
    # If tie, argmin returns first occurrence. (0,1) appears first? Index array will be row-wise: (0,1), (0,2), (1,0)... so (0,1) sum=1, (0,2) sum=2, (1,0) sum=1 -> first sum=1 is (0,1). So Min_XY = (nb=1, na=0)
    assert rec is not None
    assert isinstance( rec, dict )
    assert set( rec.keys() ) == { "Min_XY", "Min_Y", "Min_X" }
    assert rec[ "Min_XY" ] == ( 1, 0 )

    # Min_Y: smallest na with any valid, then smallest nb for that na. na=0 has valid nb=1,2 -> pick nb=1 -> (1,0)
    assert rec[ "Min_Y" ] == ( 1, 0 )

    # Min_X: smallest nb with any valid na, then smallest na. nb=0 has valid na=1 -> (0,1)
    assert rec[ "Min_X" ] == ( 0, 1 )

    # Validate grid type, shape, dtype, and values
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 3, 3 )  # (MaxLags[1]+1, MaxLags[0]+1)
    assert grid.dtype == np.float64
    assert grid[ 0, 0 ] == 0.5
    assert grid[ 0, 1 ] == 0.9
    assert grid[ 2, 2 ] == 0.95

  def test_no_config_above_threshold( self, mock_all ) -> None:
    '''None returned when no config meets threshold.'''
    mock_ctors, mock_compute_err, _, _ = mock_all
    max_lags = ( 1, 1 )
    grid_vals = { ( 0, 0 ): 0.3, ( 0, 1 ): 0.4, ( 1, 0 ): 0.5, ( 1, 1 ): 0.6 }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 1,
            MaxLags = max_lags, VarianceAcceptThreshold = 0.9, Plot = False
        )
    assert rec is None
    # Validate grid type, shape, dtype, and all values
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 2, 2 )  # (MaxLags[1]+1, MaxLags[0]+1)
    assert grid.dtype == np.float64
    assert grid[ 0, 0 ] == 0.3
    assert grid[ 0, 1 ] == 0.4
    assert grid[ 1, 0 ] == 0.5
    assert grid[ 1, 1 ] == 0.6

  def test_grid_values_clipped_at_one( self, mock_all ) -> None:
    '''ERR sums > 1.0 are clipped to 1.0 in the grid.'''
    mock_ctors, mock_compute_err, _, _ = mock_all
    max_lags = ( 1, 1 )
    # Keep (0,1) low so (1,1) is NOT skipped by the early optimization
    grid_vals = {
        ( 0, 0 ): 0.5,
        ( 0, 1 ): 0.6,  # well below 1e-11 tolerance, won't trigger skip at (1,1)
        ( 1, 0 ): 1.5,  # clipped to 1.0
        ( 1, 1 ): 0.8,
    }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    rec, grid = MaxLagsEstimator(
        tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 1,
        MaxLags = max_lags, VarianceAcceptThreshold = 0.9, Plot = False
    )
    assert grid[ 0, 0 ] == 0.5
    assert grid[ 0, 1 ] == 0.6   # not clipped
    assert grid[ 1, 0 ] == 1.0  # clipped from 1.5
    assert grid[ 1, 1 ] == 0.8
    assert rec is not None  # max=1.0 >= threshold=0.9

  def test_early_optimization_skip( self, mock_all ) -> None:
    '''Cells above threshold skip computation for larger lags.'''
    mock_ctors, mock_compute_err, _, _ = mock_all
    max_lags = ( 2, 2 )
    # Set cells (0,1), (1,0), (0,2), (2,0) ... such that for (1,1) both (0,1) and (1,0) are >= 1-1e-11.
    # We'll use exactly 1.0 to trigger skip.
    grid_vals = {
            ( 0, 0 ): 0.5,
            ( 0, 1 ): 1.0, # needed for skip (1,1)
            ( 0, 2 ): 0.0,
            ( 1, 0 ): 1.0, # needed for skip (1,1)
            ( 1, 1 ): 999, # This should never be read because of skip
            ( 1, 2 ): 0.0,
            ( 2, 0 ): 0.0,
            ( 2, 1 ): 0.0,
            ( 2, 2 ): 0.0,
        }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 1,
            MaxLags = max_lags, VarianceAcceptThreshold = 0.9, Plot = False
        )
    # (1,1) must be 1.0 (set by optimization) and ComputeERR not called for it
    assert grid[ 1, 1 ] == 1.0
    # Verify that ComputeERR was called only for cells where we expected it
    # It should not have been called with values leading to (1,1).
    # We can check call count: we expect it called for (0,0),(0,1),(1,0),(0,2),(1,2),(2,0),(2,1),(2,2) = 8 times
    assert mock_compute_err.call_count == 8
    # Validate grid type, shape, dtype, and key values
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 3, 3 )  # (MaxLags[1]+1, MaxLags[0]+1)
    assert grid.dtype == np.float64
    assert grid[ 0, 0 ] == 0.5
    assert grid[ 0, 1 ] == 1.0
    assert grid[ 1, 0 ] == 1.0
    # Cells below the skip threshold keep their computed values
    assert grid[ 2, 2 ] == 0.0

  def test_savefig_path_normalization( self, mock_all ) -> None:
    '''SaveFig path is normalized (backslash to forward slash).'''
    mock_ctors, mock_compute_err, _, mock_plt = mock_all
    max_lags = ( 1, 1 )
    grid_vals = { ( 0, 0 ): 0.9, ( 0, 1 ): 0.9, ( 1, 0 ): 0.9, ( 1, 1 ): 0.9 }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_plt.subplots.return_value = ( fig_mock, ax_mock )
    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 1,
            MaxLags = max_lags, VarianceAcceptThreshold = 0.8,
            Plot = True, SaveFig = "folder\\sub\\plot.png"
        )
    mock_plt.savefig.assert_called_once_with( "folder/sub/plot.png" )
    # Recommendations and grid should be valid
    assert rec is not None
    assert isinstance( rec, dict )
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 2, 2 )

  def test_plot_elements( self, mock_all ) -> None:
    '''Plot elements (pcolormesh, scatter, colorbar) are called.'''
    mock_ctors, mock_compute_err, _, mock_plt = mock_all
    max_lags = ( 2, 1 ) # 2 cols, 1 row -> grid shape (2,3)
    # Provide values such that max > threshold
    grid_vals = { ( 0, 0 ): 0.6, ( 0, 1 ): 0.9, ( 0, 2 ): 0.85,
                     ( 1, 0 ): 0.9, ( 1, 1 ): 0.95, ( 1, 2 ): 0.95 }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_plt.subplots.return_value = ( fig_mock, ax_mock )
    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 2,
            MaxLags = max_lags, VarianceAcceptThreshold = 0.85, Plot = True
        )
    # Check pcolormesh call
    ax_mock.pcolormesh.assert_called_once()
    call_args = ax_mock.pcolormesh.call_args
    np.testing.assert_array_equal( call_args[ 0 ][ 0 ], grid )
    assert call_args[ 1 ][ 'vmin' ] == 0.85
    expected_vmax = max( np.max( grid ), 1.0 ) # np.max(grid) = 0.95, > 0.85, so vmax=0.95
    assert call_args[ 1 ][ 'vmax' ] == 0.95

    # Scatter points for recommendations
    assert ax_mock.scatter.call_count == 3
    # Coordinates should be +0.5
    # Example: Min_Y => (nb, na) from recommendation
    # We won't hardcode the exact calls but at least verify they exist
    # Legends, colorbar, tight_layout
    ax_mock.legend.assert_not_called() # The function does not call legend for pcolormesh? Only in ExpansionOrderEstimator. In MaxLagsEstimator, there is no legend. So skip.
    fig_mock.colorbar.assert_called_once()
    fig_mock.tight_layout.assert_called()
    # Validate rec and grid types
    assert rec is not None
    assert isinstance( rec, dict )
    assert set( rec.keys() ) == { "Min_XY", "Min_Y", "Min_X" }
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 2, 3 )  # (MaxLags[1]+1, MaxLags[0]+1) = (2, 3)

  def test_plot_when_max_below_threshold_caps_colorbar( self, mock_all ) -> None:
    '''Colorbar max is capped at 1.0 when grid max below threshold.'''
    mock_ctors, mock_compute_err, _, mock_plt = mock_all
    max_lags = ( 1, 1 )
    grid_vals = { ( 0, 0 ): 0.3, ( 0, 1 ): 0.4, ( 1, 0 ): 0.5, ( 1, 1 ): 0.6 }
    self._configure_grid( mock_ctors, mock_compute_err, grid_vals, max_lags )
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_plt.subplots.return_value = ( fig_mock, ax_mock )
    rec, grid = MaxLagsEstimator(
            tor.randn( 10 ), tor.randn( 10 ), ModelOrder = 1,
            MaxLags = max_lags, VarianceAcceptThreshold = 0.9, Plot = True
        )
    # max(grid)=0.6 <= threshold => ColorBarMax = 1.0
    assert rec is None  # threshold not met
    ax_mock.pcolormesh.assert_called_once()
    assert ax_mock.pcolormesh.call_args[ 1 ][ 'vmax' ] == 1.0
    assert ax_mock.pcolormesh.call_args[ 1 ][ 'vmin' ] == 0.9
    # Validate grid type, shape, dtype, and values
    assert isinstance( grid, np.ndarray )
    assert grid.shape == ( 2, 2 )  # (MaxLags[1]+1, MaxLags[0]+1)
    assert grid.dtype == np.float64
    assert grid[ 0, 0 ] == 0.3
    assert grid[ 0, 1 ] == 0.4
    assert grid[ 1, 0 ] == 0.5
    assert grid[ 1, 1 ] == 0.6
