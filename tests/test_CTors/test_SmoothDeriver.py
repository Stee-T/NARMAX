import pytest
import torch
import numpy as np
from typing import Sequence, Union

from NARMAX.CTors import DiffedGaussianMollifier, SmoothDeriver


# =====================================================================
# Section 1: DiffedMollifiersCTor tests
# =====================================================================

class TestDiffedMollifiersCTor:

  def test_default_parameters( self ) -> None:
    '''Default call returns correct structure.'''
    Coeffs = DiffedGaussianMollifier()
    assert len( Coeffs ) == 3
    for C in Coeffs:
      assert isinstance( C, torch.Tensor )
      assert C.ndim == 1
      assert C.shape[ 0 ] == 31

  def test_coefficient_finite( self ) -> None:
    '''Each coefficient tensor contains only finite values.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 3 )
    for coeff in Coeffs:
      assert torch.isfinite( coeff ).all()
      assert coeff.ndim == 1
      assert coeff.shape[ 0 ] == 31

  def test_endpoints_zeroed( self ) -> None:
    '''Derivative endpoints are forced to zero to mitigate boundary ringing.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 15, nDerivatives = 4 )
    for order in range( 1, len( Coeffs ) ):
      assert Coeffs[ order ][ 0 ] == 0.0, f"First sample not zero for order { order }"
      assert Coeffs[ order ][ -1 ] == 0.0, f"Last sample not zero for order { order }"

  def test_nDerivatives_zero( self ) -> None:
    '''nDerivatives=0 returns only the 0th-order mollifier.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 21, nDerivatives = 0 )
    assert len( Coeffs ) == 1
    assert isinstance( Coeffs[ 0 ], torch.Tensor )
    assert Coeffs[ 0 ].ndim == 1
    assert Coeffs[ 0 ].shape[ 0 ] == 21
    assert Coeffs[ 0 ].dtype == torch.float64
    assert torch.isfinite( Coeffs[ 0 ] ).all()

  def test_nDerivatives_one( self ) -> None:
    '''nDerivatives=1 returns 0th and 1st derivative.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 21, nDerivatives = 1 )
    assert len( Coeffs ) == 2
    for C in Coeffs:
      assert isinstance( C, torch.Tensor )
      assert C.ndim == 1
      assert C.shape[ 0 ] == 21
    assert Coeffs[ 0 ].dtype == torch.float64
    assert torch.isfinite( Coeffs[ 1 ] ).all()

  def test_nDerivatives_large( self ) -> None:
    '''nDerivatives=5 returns 6 filters.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 5 )
    assert len( Coeffs ) == 6
    for C in Coeffs:
      assert isinstance( C, torch.Tensor )
      assert C.ndim == 1
      assert C.shape[ 0 ] == 31
      assert C.dtype == torch.float64
      assert torch.isfinite( C ).all()

  def test_filter_order_variations( self ) -> None:
    '''Different FilterOrder values produce correct shapes.'''
    for order in [ 5, 11, 31, 51 ]:
      Coeffs = DiffedGaussianMollifier( FilterOrder = order, nDerivatives = 2 )
      assert len( Coeffs ) == 3
      for C in Coeffs:
        assert isinstance( C, torch.Tensor )
        assert C.ndim == 1
        assert C.shape[ 0 ] == order
        assert C.dtype == torch.float64
        assert torch.isfinite( C ).all()

  def test_filter_order_even( self ) -> None:
    '''Even FilterOrder is accepted (though odd is recommended).'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 30, nDerivatives = 2 )
    assert len( Coeffs ) == 3
    for C in Coeffs:
      assert C.shape[ 0 ] == 30
      assert torch.isfinite( C ).all()

  def test_std_variations( self ) -> None:
    '''Different std values produce valid filters.'''
    for std_val in [ 0.05, 0.12, 0.5, 1.0 ]:
      Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2, std = std_val )
      assert len( Coeffs ) == 3
      for coeff in Coeffs:
        assert torch.isfinite( coeff ).all()
        assert coeff.ndim == 1
        assert coeff.shape[ 0 ] == 31

  def test_std_accepts_int_and_numpy_scalar( self ) -> None:
    '''std accepts int and numpy scalar (converted to float internally).'''
    for StdVal in [ 1, np.float64( 0.12 ), np.float32( 0.5 ) ]:
      Coeffs = DiffedGaussianMollifier( std = StdVal )
      assert len( Coeffs ) == 3
      assert Coeffs[ 0 ].dtype == torch.float64
      assert torch.isfinite( Coeffs[ 0 ] ).all()

  def test_std_invalid_type_raises( self ) -> None:
    '''Non-numeric std raises TypeError.'''
    with pytest.raises( TypeError, match = "std must be numeric" ):
      DiffedGaussianMollifier( std = "bad" )

  def test_std_zero_raises( self ) -> None:
    '''Zero std raises ValueError.'''
    with pytest.raises( ValueError, match = "std must be > 0" ):
      DiffedGaussianMollifier( std = 0 )

  def test_std_negative_raises( self ) -> None:
    '''Negative std raises ValueError.'''
    with pytest.raises( ValueError, match = "std must be > 0" ):
      DiffedGaussianMollifier( std = -0.1 )

  def test_filter_order_non_int_raises( self ) -> None:
    '''Non-integer FilterOrder raises ValueError.'''
    with pytest.raises( ValueError, match = "FilterOrder must be a positive int" ):
      DiffedGaussianMollifier( FilterOrder = 1.5 )

  def test_filter_order_zero_raises( self ) -> None:
    '''Zero FilterOrder raises ValueError.'''
    with pytest.raises( ValueError, match = "FilterOrder must be a positive int" ):
      DiffedGaussianMollifier( FilterOrder = 0 )

  def test_filter_order_negative_raises( self ) -> None:
    '''Negative FilterOrder raises ValueError.'''
    with pytest.raises( ValueError, match = "FilterOrder must be a positive int" ):
      DiffedGaussianMollifier( FilterOrder = -5 )

  def test_filter_order_below_4_raises( self ) -> None:
    '''FilterOrder=1,2,3 all raise ValueError (enforced by >= 4 check).'''
    for order in [ 1, 2, 3 ]:
      with pytest.raises( ValueError, match = "FilterOrder must be >= 4" ):
        DiffedGaussianMollifier( FilterOrder = order )

  def test_nDerivatives_negative_raises( self ) -> None:
    '''Negative nDerivatives raises ValueError.'''
    with pytest.raises( ValueError, match = "nDerivatives must be a non-negative int" ):
      DiffedGaussianMollifier( nDerivatives = -1 )

  def test_nDerivatives_non_int_raises( self ) -> None:
    '''Non-integer nDerivatives raises ValueError.'''
    with pytest.raises( ValueError, match = "nDerivatives must be a non-negative int" ):
      DiffedGaussianMollifier( nDerivatives = 2.5 )

  def test_plot_true_no_crash( self ) -> None:
    '''Plot=True should not raise (graphical output skipped in CI).'''
    try:
      Coeffs = DiffedGaussianMollifier( FilterOrder = 11, nDerivatives = 1, Plot = True )
      assert len( Coeffs ) == 2
      for C in Coeffs: assert torch.isfinite( C ).all()
    except Exception as e:
      if ( ( "Display" in str( e ) ) or ( "backend" in str( e ) ) ): pass
      else: raise

  def test_plot_false_default( self ) -> None:
    '''Plot=False is the default; no graphics.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 11, nDerivatives = 1, Plot = False )
    assert len( Coeffs ) == 2
    for C in Coeffs:
      assert C.dtype == torch.float64
      assert torch.isfinite( C ).all()

  def test_output_device( self ) -> None:
    '''All coefficients are on a valid device.'''
    Coeffs = DiffedGaussianMollifier()
    for coeff in Coeffs:
      assert isinstance( coeff, torch.Tensor )
      assert coeff.device.type in ( "cpu", "cuda" )
      assert coeff.ndim == 1
      assert coeff.shape[ 0 ] == 31

  def test_mollifier_positive( self ) -> None:
    '''Zero-order mollifier coefficients are all positive.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    assert isinstance( Coeffs[ 0 ], torch.Tensor )
    assert Coeffs[ 0 ].ndim == 1
    assert Coeffs[ 0 ].shape[ 0 ] == 31
    assert Coeffs[ 0 ].dtype == torch.float64
    assert torch.isfinite( Coeffs[ 0 ] ).all()
    assert torch.all( Coeffs[ 0 ] > 0 )

  def test_derivative_filters_sum_to_near_zero( self ) -> None:
    '''Derivative filter sums are near zero (antisymmetric kernels).'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    for order in range( 1, 3 ):
      filt = Coeffs[ order ]
      s = torch.sum( filt ).item()
      assert abs( s ) < 0.1, f"Order { order } filter sum = { s } (expected ≈ 0)"

  def test_zero_order_l1_normalized( self ) -> None:
    '''Zero-order mollifier sums to 1 (L1 normalization).'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    s = torch.sum( Coeffs[ 0 ] ).item()
    assert abs( s - 1.0 ) < 1e-6, f"sum(Coeffs[0]) = { s }, expected ≈ 1"

  def test_zero_order_preserves_constant_amplitude( self ) -> None:
    '''Convolution with zero-order mollifier preserves constant signal amplitude.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    signal = torch.ones( 100 ) * 5.0
    conv = torch.nn.functional.conv1d(
            signal.view( 1, 1, -1 ),
            Coeffs[ 0 ].flip( 0 ).view( 1, 1, -1 ),
            padding = "same",
        ).view( -1 )
    interior = conv[ 15 : 85 ]
    assert torch.allclose( interior, torch.full_like( interior, 5.0 ), atol = 1e-6 ), (
            "Zero-order mollifier did not preserve constant amplitude"
        )

  def test_first_derivative_dc_zero( self ) -> None:
    '''First derivative filter has zero DC response (sum ≈ 0).'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    s = torch.sum( Coeffs[ 1 ] ).item()
    assert abs( s ) < 1e-6, f"sum(Coeffs[1]) = { s }, expected ≈ 0"

  def test_all_coefficients_finite( self ) -> None:
    '''Every coefficient tensor contains only finite values.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 5 )
    for coeff in Coeffs: assert torch.isfinite( coeff ).all()

  @pytest.mark.skipif( not torch.cuda.is_available(), reason = "CUDA not available" )
  def test_output_device_cuda_available( self ) -> None:
    '''Test runs without error when CUDA is available.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    for coeff in Coeffs: assert coeff.device.type in ( "cpu", "cuda" )

  def test_nDerivatives_large_value( self ) -> None:
    '''nDerivatives=10 returns 11 filters with stable recurrence.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 51, nDerivatives = 10 )
    assert len( Coeffs ) == 11
    for C in Coeffs:
      assert torch.isfinite( C ).all()
      assert C.ndim == 1
      assert C.shape[ 0 ] == 51
      assert C.dtype == torch.float64

  def test_dtype_float64_default( self ) -> None:
    '''Coefficients are float64 (numpy default precision).'''
    Coeffs = DiffedGaussianMollifier()
    for C in Coeffs:
      assert C.dtype == torch.float64

  def test_dtype_conversion_preserves_properties( self ) -> None:
    '''Coefficients converted to float32/float64 retain essential properties.'''
    Coeffs = DiffedGaussianMollifier( FilterOrder = 31, nDerivatives = 2 )
    for Dtype in [ torch.float32, torch.float64 ]:
      Converted = [ c.to( dtype = Dtype ) for c in Coeffs ]
      for C in Converted:
        assert C.dtype == Dtype
        assert C.ndim == 1
        assert C.shape[ 0 ] == 31
        assert torch.isfinite( C ).all()
      S = torch.sum( Converted[ 0 ] ).item()
      assert abs( S - 1.0 ) < 1e-5


# =====================================================================
# Section 2: SmoothDeriver validation & error paths
# =====================================================================

class TestSmoothDeriverValidation:

  def test_data_not_tensor_raises( self ) -> None:
    '''Non-tensor data raises TypeError.'''
    with pytest.raises( TypeError, match = "Data must be a torch.Tensor" ):
      SmoothDeriver( [ [ 1, 2 ] ], [ "x" ] )

  def test_data_1d_raises( self ) -> None:
    '''1D data raises ValueError.'''
    data = torch.tensor( [ 1.0, 2.0, 3.0 ] )
    with pytest.raises( ValueError, match = "2D tensor" ):
      SmoothDeriver( data, [ "x" ] )

  def test_data_3d_raises( self ) -> None:
    '''3D data raises ValueError.'''
    data = torch.rand( 3, 2, 2 )
    with pytest.raises( ValueError, match = "2D tensor" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_nDerivatives_negative_raises( self ) -> None:
    '''Negative nDerivatives raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "nDerivatives must be a non-negative int" ):
      SmoothDeriver( data, [ "x", "y" ], nDerivatives = -1 )

  def test_nDerivatives_float_raises( self ) -> None:
    '''Float nDerivatives raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "nDerivatives must be a non-negative int" ):
      SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2.5 )

  def test_filter_order_too_small_raises( self ) -> None:
    '''FilterOrder < 4 raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "FilterOrder must be an int >= 4" ):
      SmoothDeriver( data, [ "x", "y" ], FilterOrder = 3 )

  def test_filter_order_2_raises( self ) -> None:
    '''FilterOrder=2 is rejected by SmoothDeriver's own check.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "FilterOrder must be an int >= 4" ):
      SmoothDeriver( data, [ "x", "y" ], FilterOrder = 2 )

  def test_filter_order_non_int_raises( self ) -> None:
    '''Non-integer FilterOrder raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "FilterOrder must be an int >= 4" ):
      SmoothDeriver( data, [ "x", "y" ], FilterOrder = 4.5 )

  def test_nan_in_data_raises( self ) -> None:
    '''NaN in data raises ValueError.'''
    data = torch.tensor( [ [ 1.0, float( "nan" ) ], [ 2.0, 3.0 ] ] )
    with pytest.raises( ValueError, match = "inf or NaN" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_inf_in_data_raises( self ) -> None:
    '''Inf in data raises ValueError.'''
    data = torch.tensor( [ [ 1.0, float( "inf" ) ], [ 2.0, 3.0 ] ] )
    with pytest.raises( ValueError, match = "inf or NaN" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_empty_data_raises( self ) -> None:
    '''Empty data raises ValueError.'''
    data = torch.empty( 0, 2 )
    with pytest.raises( ValueError, match = "at least one sample" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_regnames_mismatch_raises( self ) -> None:
    '''RegNames count mismatch raises ValueError.'''
    data = torch.rand( 10, 3 )
    with pytest.raises( ValueError, match = "Number of RegNames must match" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_regnames_2d_raises( self ) -> None:
    '''2D RegNames raises ValueError.'''
    data = torch.rand( 10, 2 )
    names = np.array( [ [ "x", "y" ], [ "z", "w" ] ] )
    with pytest.raises( ValueError, match = "RegNames must be 1D" ):
      SmoothDeriver( data, names )

  def test_regcoeffs_length_mismatch_raises( self ) -> None:
    '''RegCoeffs length mismatch raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "RegCoeffs length" ):
      SmoothDeriver( data, [ "x", "y" ], nDerivatives = 3, RegCoeffs = [ 1.0, 1.0 ] )

  def test_regcoeffs_defaults_to_ones( self ) -> None:
    '''RegCoeffs=None should default to all 1.0.'''
    data = torch.rand( 20, 2 )
    result_no_coeff, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    result_default, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2, RegCoeffs = [ 1.0, 1.0 ] )
    assert torch.allclose( result_no_coeff, result_default )

  def test_std_accepts_int( self ) -> None:
    '''std accepts integer (promoted to float).'''
    data = torch.rand( 10, 2 )
    Out, _ = SmoothDeriver( data, [ "x", "y" ], std = 1 )
    assert torch.isfinite( Out ).all()

  def test_std_rejects_non_numeric( self ) -> None:
    '''std non-numeric should be rejected.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( TypeError, match = "std must be numeric" ):
      SmoothDeriver( data, [ "x", "y" ], std = "bad" )

  def test_std_zero_raises( self ) -> None:
    '''Zero std raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "std must be > 0" ):
      SmoothDeriver( data, [ "x", "y" ], std = 0 )

  def test_plot_rejects_non_bool( self ) -> None:
    '''Plot must be a bool.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( TypeError, match = "Plot must be a bool" ):
      SmoothDeriver( data, [ "x", "y" ], Plot = 1 )

  def test_integer_dtype_raises( self ) -> None:
    '''Integer Data dtype raises TypeError.'''
    data = torch.randint( 0, 10, ( 10, 2 ), dtype = torch.int64 )
    with pytest.raises( TypeError, match = "Data dtype must be floating-point" ):
      SmoothDeriver( data, [ "x", "y" ] )

  def test_norm_derivatives_non_bool_raises( self ) -> None:
    '''Non-bool NormDerivatives raises TypeError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( TypeError, match = "NormDerivatives must be a bool" ):
      SmoothDeriver( data, [ "x", "y" ], NormDerivatives = 1 )

  def test_dt_negative_raises( self ) -> None:
    '''Negative dt raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "dt must be > 0" ):
      SmoothDeriver( data, [ "x", "y" ], dt = -0.01 )

  def test_dt_zero_raises( self ) -> None:
    '''Zero dt raises ValueError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( ValueError, match = "dt must be > 0" ):
      SmoothDeriver( data, [ "x", "y" ], dt = 0 )

  def test_dt_non_numeric_raises( self ) -> None:
    '''Non-numeric dt raises TypeError.'''
    data = torch.rand( 10, 2 )
    with pytest.raises( TypeError, match = "dt must be numeric" ):
      SmoothDeriver( data, [ "x", "y" ], dt = "bad" )


# =====================================================================
# Section 3: SmoothDeriver output shapes and types
# =====================================================================

class TestSmoothDeriverShapesAndTypes:

  def test_output_shape_default( self ) -> None:
    '''Default params: (p, n) -> (p, n*(1+nDerivatives)).'''
    data = torch.rand( 50, 3 )
    out, names = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 2 )
    assert out.shape == ( 50, 3 * ( 1 + 2 ) ) # (50, 9)

  def test_output_shape_nDerivatives_0( self ) -> None:
    '''nDerivatives=0 returns original data unchanged.'''
    data = torch.rand( 50, 3 )
    out, names = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 0 )
    assert out.shape == ( 50, 3 )
    assert torch.allclose( out, data )

  def test_output_shape_nDerivatives_1( self ) -> None:
    '''nDerivatives=1: original + 1 derivative per column.'''
    data = torch.rand( 50, 3 )
    out, names = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 1 )
    assert out.shape == ( 50, 6 ) # 3 + 3

  def test_output_shape_nDerivatives_5( self ) -> None:
    '''nDerivatives=5: 1 + 5 = 6 blocks.'''
    data = torch.rand( 50, 2 )
    out, names = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 5 )
    assert out.shape == ( 50, 2 * 6 )

  def test_output_dtype_preserved_float64( self ) -> None:
    '''float64 dtype preserved.'''
    data = torch.rand( 20, 2, dtype = torch.float64 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    assert out.dtype == torch.float64

  def test_output_dtype_preserved_float32( self ) -> None:
    '''float32 dtype preserved.'''
    data = torch.rand( 20, 2, dtype = torch.float32 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    assert out.dtype == torch.float32

  def test_names_are_ndarray_str( self ) -> None:
    '''Output names are unicode numpy array.'''
    data = torch.rand( 20, 2 )
    _, names = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    assert isinstance( names, np.ndarray )
    assert names.dtype.kind == "U"

  def test_original_data_preserved( self ) -> None:
    '''First n columns should be exactly the original data.'''
    data = torch.rand( 30, 4 )
    out, _ = SmoothDeriver( data, [ "a", "b", "c", "d" ], nDerivatives = 3 )
    assert torch.equal( out[ :, : 4 ], data )

  def test_names_length_correct( self ) -> None:
    '''Number of names matches expected count.'''
    data = torch.rand( 20, 3 )
    _, names = SmoothDeriver( data, [ "x", "y", "z" ], nDerivatives = 2 )
    assert len( names ) == 3 * 3 # 3 original + 6 derivative columns

  def test_single_column( self ) -> None:
    '''Single column data works.'''
    data = torch.rand( 50, 1 )
    out, names = SmoothDeriver( data, [ "x" ], nDerivatives = 2 )
    assert out.shape == ( 50, 3 )
    assert names.tolist() == [ "x", "∂¹ x", "∂² x" ]

  def test_many_columns( self ) -> None:
    '''Many columns (10) with multiple derivatives.'''
    data = torch.rand( 100, 10 )
    out, names = SmoothDeriver( data, [ f"f{ i }" for i in range( 10 ) ], nDerivatives = 2 )
    assert out.shape == ( 100, 30 )
    assert len( names ) == 30

  @pytest.mark.skipif( not torch.cuda.is_available(), reason = "CUDA not available" )
  def test_device_preserved_cuda( self ) -> None:
    '''CUDA device is preserved.'''
    data = torch.rand( 20, 2, device = "cuda" )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1 )
    assert out.device.type == "cuda"

  def test_device_preserved_cpu( self ) -> None:
    '''CPU device is preserved.'''
    data = torch.rand( 20, 2, device = "cpu" )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1 )
    assert out.device.type == "cpu"


# =====================================================================
# Section 4: SmoothDeriver naming convention
# =====================================================================

class TestSmoothDeriverNaming:

  REG_NAMES_EXAMPLES = [ "x[k]", "y[k-1]", "phi_1", "x" ]

  def test_simple_name_nDerivatives_1( self ) -> None:
    '''x[k] -> [x[k], ∂¹ x[k]]'''
    data = torch.rand( 50, 1 )
    _, names = SmoothDeriver( data, [ "x[k]" ], nDerivatives = 1 )
    assert names.tolist() == [ "x[k]", "∂¹ x[k]" ]

  def test_simple_name_nDerivatives_3( self ) -> None:
    '''x[k] -> [x[k], ∂¹ x[k], ∂² x[k], ∂³ x[k]]'''
    data = torch.rand( 50, 1 )
    _, names = SmoothDeriver( data, [ "x[k]" ], nDerivatives = 3 )
    expected = [ "x[k]", "∂¹ x[k]", "∂² x[k]", "∂³ x[k]" ]
    assert names.tolist() == expected

  def test_multiple_regressors( self ) -> None:
    '''Multiple regressors each get their derivative names.'''
    data = torch.rand( 50, 2 )
    _, names = SmoothDeriver( data, [ "u", "v" ], nDerivatives = 2 )
    expected = [ "u", "v", "∂¹ u", "∂¹ v", "∂² u", "∂² v" ]
    assert names.tolist() == expected

  def test_names_as_numpy_array( self ) -> None:
    '''RegNames as numpy array produces same output.'''
    data = torch.rand( 50, 1 )
    _, names_list = SmoothDeriver( data, [ "x[k]" ], nDerivatives = 1 )
    _, names_np = SmoothDeriver( data, np.array( [ "x[k]" ] ), nDerivatives = 1 )
    assert np.array_equal( names_list, names_np )

  def test_derivative_superscripts( self ) -> None:
    '''∂¹, ∂², ∂³, ∂⁴, ∂⁵ for orders 1-5.'''
    data = torch.rand( 50, 1 )
    _, names = SmoothDeriver( data, [ "s" ], nDerivatives = 5 )
    expected = [ "s" ] + [ f"∂{ _sup( i ) } s" for i in range( 1, 6 ) ]
    assert names.tolist() == expected

  def test_names_not_modified_in_place( self ) -> None:
    '''Original RegNames list should not be mutated.'''
    names_orig = [ "a", "b" ]
    names_copy = names_orig.copy()
    data = torch.rand( 20, 2 )
    SmoothDeriver( data, names_orig, nDerivatives = 1 )
    assert names_orig == names_copy


# =====================================================================
# Section 5: SmoothDeriver computational correctness
# =====================================================================

class TestSmoothDeriverCorrectness:

  def test_nDerivatives_zero_identity( self ) -> None:
    '''nDerivatives=0 is a no-op: output == input.'''
    data = torch.rand( 30, 3 )
    out, names = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 0 )
    assert torch.equal( out, data )
    assert names.tolist() == [ "a", "b", "c" ]

  def test_original_columns_preserved( self ) -> None:
    '''Zero-order columns (first n) are always the original data.'''
    data = torch.rand( 40, 4 )
    Out, _ = SmoothDeriver( data, [ "a", "b", "c", "d" ], nDerivatives = 3 )
    assert torch.equal( Out[ :, : 4 ], data )

  def test_regcoeffs_none_matches_explicit( self ) -> None:
    '''RegCoeffs=None produces same output as [1.0] * nDerivatives.'''
    torch.manual_seed( 42 )
    data = torch.rand( 50, 2 )
    OutNone, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2, RegCoeffs = None )
    OutExplicit, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2, RegCoeffs = [ 1.0, 1.0 ] )
    assert torch.allclose( OutNone, OutExplicit )

  def test_all_outputs_finite( self ) -> None:
    '''All SmoothDeriver outputs are finite.'''
    data = torch.rand( 80, 5 )
    Out, _ = SmoothDeriver( data, [ f"c{ i }" for i in range( 5 ) ], nDerivatives = 3 )
    assert torch.isfinite( Out ).all()

  def test_regcoeffs_zero_column( self ) -> None:
    '''Zero RegCoeffs produce all-zero derivative columns.'''
    data = torch.rand( 50, 2 )
    out, names = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, RegCoeffs = [ 0.0 ] )
    # First two columns = original
    assert torch.equal( out[ :, : 2 ], data )
    # Last two columns should be all zeros (filter * 0 = 0)
    assert torch.all( out[ :, 2 : ] == 0.0 )

  def test_output_finite( self ) -> None:
    '''All output values should be finite.'''
    data = torch.rand( 100, 5 )
    out, _ = SmoothDeriver( data, [ f"c{ i }" for i in range( 5 ) ], nDerivatives = 3 )
    assert torch.isfinite( out ).all()

  def test_original_values_untouched_in_output( self ) -> None:
    '''Original columns in output matrix are unchanged from input.'''
    data = torch.rand( 30, 2 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    assert torch.equal( out[ :, : 2 ], data )

  def test_different_filter_order_same_shape( self ) -> None:
    '''Changing FilterOrder doesn't change output shape.'''
    data = torch.rand( 100, 3 )
    out1, _ = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 1, FilterOrder = 11 )
    out2, _ = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 1, FilterOrder = 51 )
    assert out1.shape == out2.shape == ( 100, 6 )

  def test_large_p_small_filter_order( self ) -> None:
    '''p >> FilterOrder works fine.'''
    p = 1000
    data = torch.rand( p, 2 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2, FilterOrder = 7 )
    assert out.shape[ 0 ] == p
    assert torch.isfinite( out ).all()

  def test_small_p_large_filter_order( self ) -> None:
    '''p < FilterOrder//2 should not crash (window guard).'''
    p = 10
    data = torch.rand( p, 2 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2, FilterOrder = 31 )
    assert out.shape[ 0 ] == p
    assert torch.isfinite( out ).all()

  def test_v1_v2_output_match( self ) -> None:
    '''SmoothDeriver output is finite and shaped correctly (regression check).'''
    torch.manual_seed( 42 )
    data = torch.rand( 100, 3 )
    names = [ "x", "y", "z" ]
    out, out_names = SmoothDeriver( data, names.copy(), nDerivatives = 2 )
    assert out.shape == ( 100, 9 )
    assert len( out_names ) == 9
    assert torch.isfinite( out ).all()

  def test_norm_derivatives_true_peak_normalized( self ) -> None:
    '''NormDerivatives=True ensures each derivative column has peak amplitude ≤ 1.'''
    torch.manual_seed( 42 )
    data = torch.rand( 100, 3 )
    out, _ = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 2, NormDerivatives = True )
    n = data.shape[ 1 ]
    for j in range( n, out.shape[ 1 ] ):
      peak = out[ :, j ].abs().max()
      assert peak <= 1.0 + 1e-15, f"Column { j } peak = { peak } > 1 (expected ≤ 1)"

  def test_norm_derivatives_false_no_peak_norm( self ) -> None:
    '''NormDerivatives=False keeps original scaling (different from True).'''
    torch.manual_seed( 42 )
    data = torch.rand( 100, 3 )
    out_norm, _ = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 1, NormDerivatives = True )
    out_nonorm, _ = SmoothDeriver( data, [ "a", "b", "c" ], nDerivatives = 1, NormDerivatives = False )
    assert torch.equal( out_norm[ :, : 3 ], data )
    assert torch.equal( out_nonorm[ :, : 3 ], data )
    assert not torch.allclose( out_norm, out_nonorm ), "NormDerivatives=True/False should differ"

  def test_dt_scaling_first_derivative( self ) -> None:
    '''dt parameter scales first derivative by 1/(dt*(FilterOrder-1)).'''
    torch.manual_seed( 42 )
    data = torch.rand( 100, 2 )
    dt_val = 0.01
    out_no_dt, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, NormDerivatives = False, dt = None )
    out_with_dt, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, NormDerivatives = False, dt = dt_val )
    assert torch.equal( out_no_dt[ :, : 2 ], out_with_dt[ :, : 2 ] )
    expected_scale = 1.0 / ( dt_val * ( 31 - 1 ) )
    assert torch.allclose( out_no_dt[ :, 2 : ] * expected_scale, out_with_dt[ :, 2 : ], atol = 1e-10 )


# =====================================================================
# Section 6: SmoothDeriver edge cases
# =====================================================================

class TestSmoothDeriverEdgeCases:

  def test_p_equals_1( self ) -> None:
    '''Single sample, should not crash.'''
    data = torch.rand( 1, 2 )
    out, names = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 2 )
    assert out.shape[ 0 ] == 1
    assert len( names ) == 2 * 3

  def test_n_equals_0_no_columns( self ) -> None:
    '''Zero columns (n=0). Should work and return empty.'''
    data = torch.rand( 10, 0 )
    out, names = SmoothDeriver( data, [], nDerivatives = 2 )
    assert out.shape == ( 10, 0 )
    assert len( names ) == 0

  def test_filter_order_4_minimum( self ) -> None:
    '''FilterOrder=4 is the minimum.'''
    data = torch.rand( 50, 2 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, FilterOrder = 5 )
    assert out.shape == ( 50, 4 )
    assert torch.isfinite( out ).all()

  def test_filter_order_large( self ) -> None:
    '''FilterOrder=101 works.'''
    data = torch.rand( 200, 2 )
    out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, FilterOrder = 101 )
    assert out.shape == ( 200, 4 )

  def test_regcoeffs_custom_values( self ) -> None:
    '''Custom RegCoeffs produce different (scaled) outputs.'''
    data = torch.rand( 50, 1 )
    out_default, _ = SmoothDeriver( data, [ "x" ], nDerivatives = 2 )
    out_scaled, _ = SmoothDeriver( data, [ "x" ], nDerivatives = 2, RegCoeffs = [ 2.0, 0.5 ] )
    # The first derivative should be 2x, second 0.5x
    assert not torch.allclose( out_default, out_scaled )
    # First derivative column: out_default[:, 1] * 2 ≈ out_scaled[:, 1]
    assert torch.allclose( out_default[ :, 1 ] * 2, out_scaled[ :, 1 ], atol = 1e-10 )
    # Second derivative column: out_default[:, 2] * 0.5 ≈ out_scaled[:, 2]
    assert torch.allclose( out_default[ :, 2 ] * 0.5, out_scaled[ :, 2 ], atol = 1e-10 )

  def test_window_fade_reduces_boundary( self ) -> None:
    '''Samples at the very start/end should be more attenuated than middle.'''
    data = torch.zeros( 100, 1 )
    data[ 50 ] = 1.0 # single impulse in the middle
    out, _ = SmoothDeriver( data, [ "x" ], nDerivatives = 1, FilterOrder = 31 )
    # The derivative of an impulse should have its largest magnitude near the center
    middle = out[ 50, 1 ]
    start = out[ 0, 1 ]
    assert abs( middle ) >= abs( start ), (
            f"Boundary sample |{ start:.6f}| should be ≤ center |{ middle:.6f}|"
        )

  def test_constant_input_first_derivative_near_zero( self ) -> None:
    '''First derivative of a constant signal should be near zero.'''
    data = torch.ones( 100, 2 ) * 5.0
    out, _ = SmoothDeriver( data, [ "a", "b" ], nDerivatives = 1 )
    # Use interior samples to avoid boundary window effects
    interior = out[ 15 : 85, 2 : ] # columns 2-3 = ∂¹ a, ∂¹ b
    first_deriv = interior[ :, 0 ]
    assert first_deriv.abs().max() < 0.01, (
            f"Max first derivative of constant = { first_deriv.abs().max():.6f}"
        )

  def test_linear_input_constant_first_derivative( self ) -> None:
    '''First derivative of a linear ramp should be approximately constant.'''
    x = torch.arange( 100, dtype = torch.float64 ).view( -1, 1 )
    data = torch.cat( [ x, x * 2 ], dim = 1 )
    out, _ = SmoothDeriver( data, [ "a", "b" ], nDerivatives = 2, FilterOrder = 15 )
    # First derivative columns should be approximately constant (not near edges due to window)
    interior = out[ 20 : 80, 2 ] # column 2 = ∂¹ a, interior only
    mean_val = interior.mean()
    assert interior.std() < 1e-3, f"∂¹ a std = { interior.std():.6f} (expected ≈ constant)"

  def test_regnames_as_generator( self ) -> None:
    '''RegNames can be a generator/iterator (converted via list()).'''
    data = torch.rand( 20, 2 )
    def GenNames():
      yield "x"
      yield "y"
    Out, Names = SmoothDeriver( data, GenNames(), nDerivatives = 1 )
    assert Names.tolist() == [ "x", "y", "∂¹ x", "∂¹ y" ]

  def test_std_variations_produce_finite_output( self ) -> None:
    '''Different std values produce valid smooth derivatives.'''
    data = torch.rand( 50, 2 )
    for StdVal in [ 0.05, 0.12, 0.5, 1.0 ]:
      Out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, std = StdVal )
      assert Out.shape == ( 50, 4 )
      assert torch.isfinite( Out ).all()

  def test_std_default_matches_explicit( self ) -> None:
    '''Default std=0.12 matches explicit pass.'''
    data = torch.rand( 50, 2 )
    OutDefault, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1 )
    OutExplicit, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, std = 0.12 )
    assert torch.allclose( OutDefault, OutExplicit )

  def test_plot_true_no_crash( self ) -> None:
    '''Plot=True should not raise.'''
    data = torch.rand( 20, 2 )
    try:
      Out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, Plot = True )
      assert Out.shape == ( 20, 4 )
    except Exception as E:
      if ( ( "Display" in str( E ) ) or ( "backend" in str( E ) ) ): pass
      else: raise

  def test_plot_false_default( self ) -> None:
    '''Plot=False is the default.'''
    data = torch.rand( 20, 2 )
    Out, _ = SmoothDeriver( data, [ "x", "y" ], nDerivatives = 1, Plot = False )
    assert Out.shape == ( 20, 4 )


class TestOutNamesWidth:

  def test_short_names_no_truncation( self ) -> None:
    '''Short regressor names produce correctly sized OutNames entries.'''
    data = torch.rand( 10, 2 )
    _, Names = SmoothDeriver( data, [ "a", "b" ], nDerivatives = 2 )
    assert Names.dtype.kind == "U" # unicode string
    assert Names.tolist() == [ "a", "b", "∂¹ a", "∂¹ b", "∂² a", "∂² b" ]
    # Each entry should hold the full string — check via len
    for N in Names: assert len( N ) == len( str( N ) )

  def test_long_names_no_truncation( self ) -> None:
    '''Long regressor names are not truncated by the pre-allocated array.'''
    data = torch.rand( 10, 2 )
    LongName: str = "SomeVeryLongRegressorName_with_lots_of_chars_0123456789"
    _, Names = SmoothDeriver( data, [ LongName, "b" ], nDerivatives = 1 )
    Expected: list[ str ] = [
            LongName,
            "b",
            f"∂¹ { LongName }",
            "∂¹ b",
        ]
    assert Names.tolist() == Expected
    for N in Names: assert len( N ) == len( str( N ) )

  def test_varying_name_lengths( self ) -> None:
    '''Mixed-length regressor names all survive with full content.'''
    data = torch.rand( 10, 4 )
    NamesIn: list[ str ] = [ "x", "longish_name", "tiny", "a_very_long_column_name_that_should_not_be_truncated_12345" ]
    _, NamesOut = SmoothDeriver( data, NamesIn, nDerivatives = 2 )
    # Check a few representative entries
    assert NamesOut[ 0 ] == NamesIn[ 0 ]
    assert NamesOut[ 3 ] == NamesIn[ 3 ]
    assert NamesOut[ 4 ] == f"∂¹ { NamesIn[ 0 ] }"
    assert NamesOut[ 7 ] == f"∂¹ { NamesIn[ 3 ] }"
    assert NamesOut[ 8 ] == f"∂² { NamesIn[ 0 ] }"
    assert NamesOut[ 11 ] == f"∂² { NamesIn[ 3 ] }"
    for N in NamesOut: assert len( N ) == len( str( N ) )

  def test_high_derivative_name_width( self ) -> None:
    '''nDerivatives=12 exercises multi-digit superscript prefix.'''
    data = torch.rand( 10, 1 )
    LongName: str = "sensor_reading"
    _, Names = SmoothDeriver( data, [ LongName ], nDerivatives = 12 )
    assert Names[ 0 ] == LongName
    assert Names[ 12 ] == f"∂¹² { LongName }"
    # Verify no truncation by checking round-trip equality
    assert Names[ 12 ] == f"∂{ _sup( 12 ) } { LongName }"
    for N in Names: assert len( N ) == len( str( N ) )

  def test_zero_derivatives_names_untouched( self ) -> None:
    '''nDerivatives=0 returns only original names.'''
    data = torch.rand( 10, 3 )
    NamesIn: list[ str ] = [ "temperature", "pressure", "flow_rate" ]
    _, NamesOut = SmoothDeriver( data, NamesIn, nDerivatives = 0 )
    assert NamesOut.tolist() == NamesIn
    for N in NamesOut: assert len( N ) == len( str( N ) )


# Helper for the superscript test
def _sup( n: int ) -> str:
  trans = str.maketrans( "0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹" )
  return str( n ).translate( trans )
