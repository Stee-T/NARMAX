import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from NARMAX.Tools import IIR_Spectrum

@pytest.fixture
def mock_plot_funcs() -> tuple[ MagicMock, MagicMock, MagicMock, MagicMock, MagicMock ]:
  '''Mocks matplotlib and scipy functions used by IIR_Spectrum.'''
  with patch( 'NARMAX.Tools.plt.subplots' ) as mock_subplots, patch( 'NARMAX.Tools.sps.freqz' ) as mock_freqz, patch( 'NARMAX.Tools.plt.style.use', return_value = None ): # suppress dark_background
    # Setup a mock figure and two axes
    mock_fig = MagicMock( name = 'fig' )
    mock_ax0 = MagicMock( name = 'ax0' )
    mock_ax1 = MagicMock( name = 'ax1' )
    mock_subplots.return_value = ( mock_fig, ( mock_ax0, mock_ax1 ) )
    yield mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1


class TestIIRSpectrum:
  '''Tests for IIR_Spectrum frequency response plotting.'''

  # ---------------------------------------------------------------
  # Basic plotting with b_a coefficients
  # ---------------------------------------------------------------
  def test_single_ba_filter_auto_ylim( self, mock_plot_funcs ) -> None:
    '''Single b/a filter with automatic y-limits.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    b = [ 0.1, 0.2 ]
    a = [ 1.0, -0.5 ]
    # freqz returns w, h
    mock_freqz.return_value = ( np.array( [ 0, 10, 20 ] ), np.array( [ 1 + 0j, 0.5 + 0.5j, 0.2 - 0.3j ] ) )
    fig, ax = IIR_Spectrum( b_a_List = [ ( b, a ) ], FilterNames = [ 'Test' ] )
    # Check freqz call
    mock_freqz.assert_called_once_with( b, a, worN = 5000, fs = 44100 )
    # semilogx calls: both axes should have been called with mask w>0
    assert mock_ax0.semilogx.called
    assert mock_ax1.semilogx.called
    # Magnitude data should be 20*log10(max(abs(h),1e-6))
    expected_mag = 20 * np.log10( np.maximum( np.abs( [ 0.5 + 0.5j, 0.2 - 0.3j ] ), 1e-06 ) )
    # Phase unwrap
    expected_phase = np.unwrap( np.angle( [ 0.5 + 0.5j, 0.2 - 0.3j ] ) )
    # Check that semilogx was called with correct data (w>0 mask)
    w_positive = np.array( [ 10, 20 ] )
    args_mag, _ = mock_ax0.semilogx.call_args
    np.testing.assert_array_almost_equal( args_mag[ 0 ], w_positive )
    np.testing.assert_array_almost_equal( args_mag[ 1 ], expected_mag )
    args_phase, _ = mock_ax1.semilogx.call_args
    np.testing.assert_array_almost_equal( args_phase[ 0 ], w_positive )
    np.testing.assert_array_almost_equal( args_phase[ 1 ], expected_phase )
    # ylim auto: lower should be min-2, upper max+2
    # Note: yLimMag is computed from ALL elements of h (including DC)
    all_mag = 20 * np.log10( np.maximum( np.abs( mock_freqz.return_value[ 1 ] ), 1e-06 ) )
    ylim_low = np.min( all_mag ) - 2
    ylim_high = np.max( all_mag ) + 2
    mock_ax0.set_ylim.assert_called_once_with( ylim_low, ylim_high )

  def test_multiple_ba_filters_legend( self, mock_plot_funcs ) -> None:
    '''Multiple b/a filters show legend.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.side_effect = [
            ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) ),
            ( np.array( [ 1, 2 ] ), np.array( [ 0.5, 0.5 ] ) )
        ]
    b_a = [ ( [ 0.1 ], [ 1.0 ] ), ( [ 0.2 ], [ 1.0 ] ) ]
    IIR_Spectrum( b_a_List = b_a, FilterNames = [ 'A', 'B' ] )
    assert mock_ax0.semilogx.call_count == 2
    assert mock_ax1.semilogx.call_count == 2
    mock_ax0.legend.assert_called_once_with( [ 'A', 'B' ] )
    mock_ax1.legend.assert_called_once_with( [ 'A', 'B' ] )

  # ---------------------------------------------------------------
  # Plotting with h_List (no freqz)
  # ---------------------------------------------------------------
  def test_h_list_no_freqz( self, mock_plot_funcs ) -> None:
    '''h_List input skips freqz call.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    # Note: w = linspace(0, Fs/2, Resolution) so h must match Resolution length
    h = [ 0.5 + 0.5j, 0.2 - 0.3j ]
    IIR_Spectrum( h_List = [ h ], Fs = 1000, Resolution = len( h ) )
    mock_freqz.assert_not_called()
    # semilogx must have been called with w>0 (default linspace)
    expected_w = np.linspace( 0, 500, 2, endpoint = True ) # Fs/2 = 500
    mask = expected_w > 0
    w_positive = expected_w[ mask ]
    h_arr = np.array( h, dtype = complex )
    expected_mag = 20 * np.log10( np.maximum( np.abs( h_arr ), 1e-06 ) )
    expected_phase = np.unwrap( np.angle( h_arr ) )
    args_mag, _ = mock_ax0.semilogx.call_args
    np.testing.assert_array_equal( args_mag[ 0 ], w_positive )
    np.testing.assert_array_almost_equal( args_mag[ 1 ], expected_mag[ mask ] )
    args_phase, _ = mock_ax1.semilogx.call_args
    np.testing.assert_array_equal( args_phase[ 0 ], w_positive )
    np.testing.assert_array_almost_equal( args_phase[ 1 ], expected_phase[ mask ] )

  def test_h_list_with_custom_w( self, mock_plot_funcs ) -> None:
    '''Custom w_List is used directly.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    w = np.array( [ 10, 20, 30 ] )
    h = np.array( [ 1, 2, 3 ], dtype = complex )
    IIR_Spectrum( h_List = [ h ], w_List = w, Fs = 100 )
    mock_freqz.assert_not_called()
    # Should have used w directly, skipping mask for w>0 if w contains 0?
    # Our implementation masks w>0; w is all positive, so fine.
    expected_mag = 20 * np.log10( np.maximum( np.abs( h ), 1e-06 ) )
    expected_phase = np.unwrap( np.angle( h ) )
    args_mag, _ = mock_ax0.semilogx.call_args
    np.testing.assert_array_equal( args_mag[ 0 ], w )
    np.testing.assert_array_almost_equal( args_mag[ 1 ], expected_mag )
    args_phase, _ = mock_ax1.semilogx.call_args
    np.testing.assert_array_equal( args_phase[ 0 ], w )
    np.testing.assert_array_almost_equal( args_phase[ 1 ], expected_phase )

  # ---------------------------------------------------------------
  # yLimMag behaviour
  # ---------------------------------------------------------------
  def test_manual_ylim_mag_respected( self, mock_plot_funcs ) -> None:
    '''Manual yLimMag values are respected.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 0.1, 0.05 ] ) )
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], yLimMag = [ -40, 10 ] )
    mock_ax0.set_ylim.assert_called_once_with( -40, 10 )

  def test_manual_ylim_mag_with_none_low( self, mock_plot_funcs ) -> None:
    '''Manual yLimMag with None low uses auto lower bound.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 0.1 ] ) )
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], yLimMag = [ None, 0 ] )
    # low should be auto (min-2), high fixed at 0
    mag = 20 * np.log10( np.maximum( np.abs( [ 1, 0.1 ] ), 1e-6 ) )
    expected_low = np.min( mag ) - 2
    mock_ax0.set_ylim.assert_called_once_with( expected_low, 0 )

  def test_auto_ylim_sanity_check_fails( self, mock_plot_funcs ) -> None:
    '''Invalid manual limits raise ValueError.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) )
    with pytest.raises( ValueError, match = "Final magnitude limits are invalid" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], yLimMag = [ 10, 5 ] )

  # ---------------------------------------------------------------
  # Input validation errors
  # ---------------------------------------------------------------
  def test_both_none_raises( self, mock_plot_funcs ) -> None:
    '''Both arguments None raises ValueError.'''
    with pytest.raises( ValueError, match = "Either b_a_List or h_List" ):
      IIR_Spectrum()

  def test_both_provided_raises( self, mock_plot_funcs ) -> None:
    '''Both b_a_List and h_List provided raises ValueError.'''
    with pytest.raises( ValueError, match = "Either b_a_List or h_List" ):
      IIR_Spectrum( b_a_List = [], h_List = [] )

  def test_empty_coefflist_raises( self, mock_plot_funcs ) -> None:
    '''Empty b_a_List raises ValueError.'''
    with pytest.raises( ValueError, match = "must contain at least one filter" ):
      IIR_Spectrum( b_a_List = [] )

  def test_b_a_not_list_of_2tuples_raises( self, mock_plot_funcs ) -> None:
    '''b_a_List not list of 2-tuples raises ValueError.'''
    with pytest.raises( ValueError, match = "list of 2D-tuples" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], ) ] ) # only one element

  def test_h_list_not_arrays_raises( self, mock_plot_funcs ) -> None:
    '''h_List not array-like raises ValueError.'''
    with pytest.raises( ValueError, match = "list of numpy arrays" ):
      IIR_Spectrum( h_List = [ 42 ] ) # int not ndarray/iterable

  def test_invalid_Fs_raises( self, mock_plot_funcs ) -> None:
    '''Invalid Fs values raise ValueError.'''
    with pytest.raises( ValueError, match = "Fs must be an integer or float" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Fs = '44k' )
    with pytest.raises( ValueError, match = "Fs must be a positive integer" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Fs = 0 )

  def test_resolution_negative_raises( self, mock_plot_funcs ) -> None:
    '''Non-positive Resolution raises ValueError.'''
    with pytest.raises( ValueError, match = "Resolution must be a positive integer" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Resolution = 0 )

  def test_filternames_length_mismatch_raises( self, mock_plot_funcs ) -> None:
    '''FilterNames length mismatch raises ValueError.'''
    with pytest.raises( ValueError, match = "CoeffList and FilterNames must have the same length" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], FilterNames = [ 'A', 'B' ] )

  def test_w_list_not_1d_raises( self, mock_plot_funcs ) -> None:
    '''Non-1D w_List raises ValueError.'''
    with pytest.raises( ValueError, match = "w_List must be a 1D array" ):
      IIR_Spectrum( h_List = [ [ 1 ] ], w_List = [ [ 1, 2 ] ], Fs = 100 )

  def test_w_list_out_of_range_raises( self, mock_plot_funcs ) -> None:
    '''w_List values out of range raise ValueError.'''
    with pytest.raises( ValueError, match = "w_List values must be between 0 and Fs/2" ):
      IIR_Spectrum( h_List = [ [ 1, 2 ] ], w_List = [ 0, 60 ], Fs = 100 ) # 60 > 50

  def test_h_list_length_mismatch_with_w_raises( self, mock_plot_funcs ) -> None:
    '''h_List length mismatch with w_List raises ValueError.'''
    h = np.array( [ 1, 2, 3 ] )
    w = np.array( [ 10, 20 ] )
    with pytest.raises( ValueError, match = "does not match w_List length" ):
      IIR_Spectrum( h_List = [ h ], w_List = w, Fs = 100 )

  # ---------------------------------------------------------------
  # Return types and figure/axes configuration
  # ---------------------------------------------------------------
  def test_return_types( self, mock_plot_funcs ) -> None:
    '''Returns (Figure, ndarray-like) with correct elements.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) )
    fig, ax = IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ] )
    assert fig is mock_fig
    assert len( ax ) == 2
    assert ax[ 0 ] is mock_ax0
    assert ax[ 1 ] is mock_ax1

  def test_default_filter_names( self, mock_plot_funcs ) -> None:
    '''Default FilterNames when not provided.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.side_effect = [
      ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) ),
      ( np.array( [ 1, 2 ] ), np.array( [ 0.5, 0.5 ] ) )
    ]
    IIR_Spectrum( b_a_List = [ ( [ 0.1 ], [ 1.0 ] ), ( [ 0.2 ], [ 1.0 ] ) ] )
    mock_ax0.legend.assert_called_once_with( [ 'Filter 1', 'Filter 2' ] )
    mock_ax1.legend.assert_called_once_with( [ 'Filter 1', 'Filter 2' ] )

  def test_default_xlims( self, mock_plot_funcs ) -> None:
    '''Default xLims is [1, Fs/2].'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) )
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Fs = 1000 )
    mock_ax0.set_xlim.assert_called_once_with( [ 1, 500.0 ] )

  def test_axes_setup_calls( self, mock_plot_funcs ) -> None:
    '''Title, labels, and grid are set on both axes.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) )
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ] )
    mock_ax0.set_title.assert_called_once_with( 'Frequency Response' )
    mock_ax0.set_xlabel.assert_called_once_with( 'Frequency [Hz]' )
    mock_ax0.set_ylabel.assert_called_once_with( 'Magnitude [dB]' )
    mock_ax1.set_xlabel.assert_called_once_with( 'Frequency [Hz]' )
    mock_ax1.set_ylabel.assert_called_once_with( 'Phase [Radians]' )
    mock_ax0.grid.assert_called_once_with( which = 'both', alpha = 0.2 )
    mock_ax1.grid.assert_called_once_with( which = 'both', alpha = 0.2 )

  def test_tight_layout_called( self, mock_plot_funcs ) -> None:
    '''tight_layout is called on the figure.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.return_value = ( np.array( [ 1, 2 ] ), np.array( [ 1, 1 ] ) )
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ] )
    mock_fig.tight_layout.assert_called_once()

  # ---------------------------------------------------------------
  # Additional input validation edge cases
  # ---------------------------------------------------------------
  def test_h_list_empty_raises( self, mock_plot_funcs ) -> None:
    '''Empty h_List raises ValueError.'''
    with pytest.raises( ValueError, match = "must contain at least one filter" ):
      IIR_Spectrum( h_List = [] )

  def test_h_list_length_mismatch_resolution_raises( self, mock_plot_funcs ) -> None:
    '''h_List length mismatch with Resolution raises ValueError.'''
    h = np.array( [ 1, 2, 3 ] )
    with pytest.raises( ValueError, match = "does not match Resolution" ):
      IIR_Spectrum( h_List = [ h ], Fs = 100, Resolution = 5 )

  def test_fs_negative_raises( self, mock_plot_funcs ) -> None:
    '''Negative Fs raises ValueError.'''
    with pytest.raises( ValueError, match = "Fs must be a positive integer" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Fs = -1 )
    with pytest.raises( ValueError, match = "Fs must be a positive integer" ):
      IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ) ], Fs = -100.5 )

  def test_coefflist_not_list_or_tuple_raises( self, mock_plot_funcs ) -> None:
    '''CoeffList not a list or tuple raises ValueError.'''
    with pytest.raises( ValueError, match = "CoeffList must be a list or tuple" ):
      IIR_Spectrum( b_a_List = 'not a list' )

  # ---------------------------------------------------------------
  # Auto yLimMag across multiple filters
  # ---------------------------------------------------------------
  def test_auto_ylim_two_filters( self, mock_plot_funcs ) -> None:
    '''Auto yLimMag tracks min/max across multiple filters.'''
    mock_subplots, mock_freqz, mock_fig, mock_ax0, mock_ax1 = mock_plot_funcs
    mock_freqz.side_effect = [
      ( np.array( [ 1, 2 ] ), np.array( [ 1.0, 0.5 ] ) ),   # mag ≈ [0, -6.02] dB
      ( np.array( [ 1, 2 ] ), np.array( [ 0.1, 0.01 ] ) ),  # mag ≈ [-20, -40] dB
    ]
    IIR_Spectrum( b_a_List = [ ( [ 1 ], [ 1 ] ), ( [ 0.1 ], [ 1 ] ) ] )
    mag1 = 20 * np.log10( np.maximum( np.abs( [ 1.0, 0.5 ] ), 1e-06 ) )
    mag2 = 20 * np.log10( np.maximum( np.abs( [ 0.1, 0.01 ] ), 1e-06 ) )
    all_mag = np.concatenate( [ mag1, mag2 ] )
    expected_low = np.min( all_mag ) - 2
    expected_high = np.max( all_mag ) + 2
    mock_ax0.set_ylim.assert_called_once_with( expected_low, expected_high )
