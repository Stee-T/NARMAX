import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from NARMAX.Tools import zPlanePlot

@pytest.fixture
def mock_plot_funcs() -> tuple[ MagicMock, MagicMock, MagicMock, MagicMock ]:
  with patch( 'NARMAX.Tools.plt.subplots' ) as mock_subplots, patch( 'NARMAX.Tools.patches.Circle' ) as mock_circle, patch( 'NARMAX.Tools.plt.style.use', return_value = None ):
    mock_fig = MagicMock( name = 'fig' )
    mock_ax = MagicMock( name = 'ax' )
    mock_subplots.return_value = ( mock_fig, mock_ax )
    # Make add_patch return something so we can check calls
    mock_circle.return_value = MagicMock( name = 'circle_patch' )
    yield mock_subplots, mock_fig, mock_ax, mock_circle


class TestZPlanePlot:
  '''Tests for zPlanePlot – pole-zero plot.'''

  # ---------------------------------------------------------------
  # Core computations (z, p, k) are returned correctly
  # ---------------------------------------------------------------
  def test_simple_filter( self, mock_plot_funcs ) -> None:
    '''Simple filter returns correct zeros, poles, and gain.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -2 ] # zero at z=2
    a = [ 1, -0.5 ] # pole at z=0.5
    z, p, k = zPlanePlot( b, a )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    assert z.ndim == 1
    assert p.ndim == 1
    np.testing.assert_array_almost_equal( z, np.roots( b ) )
    np.testing.assert_array_almost_equal( p, np.roots( a ) )
    assert k == b[ 0 ] / a[ 0 ] == 1.0

  def test_scalar_a_default( self, mock_plot_funcs ) -> None:
    '''Default a=1 works correctly (no poles).'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 2, 3 ] # zero at -1.5
    z, p, k = zPlanePlot( b )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    assert z.ndim == 1
    assert p.ndim == 1
    assert p.size == 0
    np.testing.assert_array_almost_equal( z, np.roots( b ) )
    np.testing.assert_array_almost_equal( p, np.roots( [ 1 ] ) ) # a=1 -> no poles
    assert k == b[ 0 ] / 1.0

  def test_zero_leading_denominator_raises( self, mock_plot_funcs ) -> None:
    '''a[0]=0 raises ValueError.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    with pytest.raises( ValueError, match = r"Denominator leading coefficient a\[0\] must not be zero" ):
      zPlanePlot( b = [ 1 ], a = [ 0, 1 ] )

  # ---------------------------------------------------------------
  # Plotting commands
  # ---------------------------------------------------------------
  def test_unit_circle_added( self, mock_plot_funcs ) -> None:
    '''Unit circle is added as a dashed patch.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -0.5 ]
    zPlanePlot( b )
    # Circle should have been created with radius=1, fill=False, ls='dashed'
    mock_circle.assert_called_once_with( ( 0, 0 ), radius = 1, fill = False, ls = 'dashed' )
    mock_ax.add_patch.assert_called_once_with( mock_circle.return_value )
    mock_ax.axis.assert_called_with( 'scaled' )
    mock_ax.grid.assert_called_once_with( which = 'both', alpha = 0.15 )

  def test_zeros_and_poles_plotted( self, mock_plot_funcs ) -> None:
    '''Zeros and poles are plotted with correct markers.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -1, 0.5 ] # two zeros
    a = [ 1, -0.5, 0.2 ] # two poles
    zPlanePlot( b, a )
    # Check that ax.plot was called twice: once for zeros, once for poles
    calls = mock_ax.plot.call_args_list
    assert len( calls ) == 2
    # First call: zeros (green 'go', ms=10)
    args1, kwargs1 = calls[ 0 ]
    np.testing.assert_array_almost_equal( args1[ 0 ], np.real( np.roots( b ) ) )
    np.testing.assert_array_almost_equal( args1[ 1 ], np.imag( np.roots( b ) ) )
    assert args1[ 2 ] == 'go'
    assert kwargs1.get( 'ms' ) == 10
    # Second call: poles (red 'rx', ms=10)
    args2, kwargs2 = calls[ 1 ]
    np.testing.assert_array_almost_equal( args2[ 0 ], np.real( np.roots( a ) ) )
    np.testing.assert_array_almost_equal( args2[ 1 ], np.imag( np.roots( a ) ) )
    assert args2[ 2 ] == 'rx'
    assert kwargs2.get( 'ms' ) == 10
    mock_ax.axis.assert_called_with( 'scaled' )
    mock_ax.grid.assert_called_once_with( which = 'both', alpha = 0.15 )

  def test_axis_limits_include_unit_circle( self, mock_plot_funcs ) -> None:
    '''Axis limits include the unit circle with padding.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    # all poles/zeros inside unit circle -> Lim = 1.0 + 0.1 = 1.1
    b = [ 1, -0.5 ]
    a = [ 1, 0.2 ]
    zPlanePlot( b, a )
    mock_ax.set_xlim.assert_called_with( -1.1, 1.1 )
    mock_ax.set_ylim.assert_called_with( -1.1, 1.1 )
    mock_ax.axis.assert_called_with( 'scaled' )

  def test_axis_limits_extend_beyond_circle( self, mock_plot_funcs ) -> None:
    '''Axis limits extend beyond the unit circle when zeros/poles are outside.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    # zero at 3.0 -> Lim = max(1,3)+0.1 = 3.1
    b = [ 1, -3 ]
    zPlanePlot( b )
    mock_ax.set_xlim.assert_called_with( -3.1, 3.1 )
    mock_ax.set_ylim.assert_called_with( -3.1, 3.1 )
    mock_ax.axis.assert_called_with( 'scaled' )

  def test_tick_spacing_large_range( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 100 for range > 100.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -101 ] # zero at 101, Lim=101.1 -> TickSpacing=100
    zPlanePlot( b )
    # Check that set_xticks was called with appropriate arange
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    # Should be array from -200 to 200 step 100
    expected = np.arange( -200, 201, 100 )
    np.testing.assert_array_equal( call_args, expected )
    # set_yticks should match set_xticks
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_50_to_100( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 10 for range 50-100.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -54.9 ] # zero at 54.9, Lim=55.0 -> TickSpacing=10
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -60, 70, 10 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_25_to_50( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 5 for range 25-50.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -29.9 ] # zero at 29.9, Lim=30.0 -> TickSpacing=5
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -35, 40, 5 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_10_to_25( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 2 for range 10-25.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -14.9 ] # zero at 14.9, Lim=15.0 -> TickSpacing=2
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -16, 18, 2 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_5_to_10( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 1 for range 5-10.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -6.9 ] # zero at 6.9, Lim=7.0 -> TickSpacing=1
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -8, 9, 1 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_1_5_to_5( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 0.5 for range 1.5-5.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -1.9 ] # zero at 1.9, Lim=2.0 -> TickSpacing=0.5
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -2.5, 3.0, 0.5 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_tick_spacing_below_1_5( self, mock_plot_funcs ) -> None:
    '''Tick spacing is 0.25 for range < 1.5.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1, -1.1 ] # zero at 1.1, Lim=1.2 -> TickSpacing=0.25
    zPlanePlot( b )
    call_args = mock_ax.set_xticks.call_args[ 0 ][ 0 ]
    expected = np.arange( -1.25, 1.5, 0.25 )
    np.testing.assert_array_equal( call_args, expected )
    ycall_args = mock_ax.set_yticks.call_args[ 0 ][ 0 ]
    np.testing.assert_array_equal( ycall_args, expected )

  def test_title_passed( self, mock_plot_funcs ) -> None:
    '''Title is passed to suptitle.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    zPlanePlot( [ 1, -0.5 ], Title = "My Filter" )
    mock_fig.suptitle.assert_called_once_with( "My Filter" )

  def test_no_title_when_none( self, mock_plot_funcs ) -> None:
    '''No title when Title is None.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    zPlanePlot( [ 1, -0.5 ] )
    mock_fig.suptitle.assert_not_called()

  def test_returns_zpk( self, mock_plot_funcs ) -> None:
    '''Returns correct zeros, poles, and gain.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 2, 1 ]
    a = [ 3, -0.5 ]
    z, p, k = zPlanePlot( b, a )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    np.testing.assert_array_almost_equal( z, np.roots( b ) )
    np.testing.assert_array_almost_equal( p, np.roots( a ) )
    assert k == 2 / 3

  def test_empty_b_raises( self, mock_plot_funcs ) -> None:
    '''Empty b raises IndexError.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    with pytest.raises( IndexError, match = r"index 0 is out of bounds for axis 0 with size 0" ):
      zPlanePlot( b = [], a = [ 1 ] )

  def test_empty_a_raises( self, mock_plot_funcs ) -> None:
    '''Empty a raises IndexError.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    with pytest.raises( IndexError, match = r"index 0 is out of bounds for axis 0 with size 0" ):
      zPlanePlot( b = [ 1 ], a = [] )

  # ---------------------------------------------------------------
  # Additional edge cases and configuration checks
  # ---------------------------------------------------------------
  def test_tight_layout_called( self, mock_plot_funcs ) -> None:
    '''tight_layout is called on the figure.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    zPlanePlot( [ 1, -0.5 ] )
    mock_fig.tight_layout.assert_called_once()

  def test_spine_configuration( self, mock_plot_funcs ) -> None:
    '''Spines are configured to center at zero and hide top/right.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    zPlanePlot( [ 1, -0.5 ] )
    mock_ax.spines[ 'bottom' ].set_position.assert_called_with( 'zero' )
    mock_ax.spines[ 'left' ].set_position.assert_called_with( 'zero' )
    mock_ax.spines[ 'top' ].set_visible.assert_called_with( False )
    mock_ax.spines[ 'right' ].set_visible.assert_called_with( False )

  def test_complex_conjugate_poles( self, mock_plot_funcs ) -> None:
    '''Complex conjugate poles are returned correctly.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1 ]
    a = [ 1, 0, 1 ] # poles at +j and -j
    z, p, k = zPlanePlot( b, a )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    assert p.size == 2
    np.testing.assert_array_almost_equal( p, [ 1j, -1j ] )
    assert k == 1.0

  def test_integer_inputs( self, mock_plot_funcs ) -> None:
    '''Integer numerator/denominator inputs work correctly.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 2, -4 ]
    a = [ 4, -2 ]
    z, p, k = zPlanePlot( b, a )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    np.testing.assert_array_almost_equal( z, np.roots( [ 2, -4 ] ) )
    np.testing.assert_array_almost_equal( p, np.roots( [ 4, -2 ] ) )
    assert k == 0.5

  def test_only_poles( self, mock_plot_funcs ) -> None:
    '''Only poles (b = [1]) returns no zeros.'''
    mock_subplots, mock_fig, mock_ax, mock_circle = mock_plot_funcs
    b = [ 1 ]
    a = [ 1, -0.5, 0.1 ]
    z, p, k = zPlanePlot( b, a )
    assert isinstance( z, np.ndarray )
    assert isinstance( p, np.ndarray )
    assert isinstance( k, ( float, np.floating ) )
    assert z.size == 0
    assert p.size == 2
    np.testing.assert_array_almost_equal( p, np.roots( a ) )
    assert k == 1.0
