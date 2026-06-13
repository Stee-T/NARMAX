# test_init_and_compute_buffer.py
import pytest
import torch
import copy
from typing import List, Optional

# Adjust the import to your actual package structure
from NARMAX import SymbolicOscillator, NonLinearity
from NARMAX.Validation import InitAndComputeBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_dummy_nonlinearity() -> list[ NonLinearity ]:
  '''Return a list with one unused NonLinearity (constructor requires at least one).'''
  return [ NonLinearity( "dummy", f = lambda x : x ) ]


def create_model(
    y_lags: List[ int ],
    x_lags: List[ int ],
    theta: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SymbolicOscillator:
  '''
  Build a SymbolicOscillator with the given output and input lags.
  Expressions are added in the order: first all y terms, then all x terms.
  theta must have length = len(y_lags) + len(x_lags).
  '''
  # OutputVarName is excluded from InputVarName2Idx, so ordering no longer matters
  input_vars = [ "y", "x" ]
  # Expressions
  exprs = []
  for lag in y_lags: exprs.append( f"y[k-{ lag }]" )
  for lag in x_lags:
    if ( lag == 0 ): exprs.append( "x[k]" )
    else: exprs.append( f"x[k-{ lag }]" )

  # ModelVarNames must include both 'y' (output var) and 'x' (input var)
  return SymbolicOscillator(
        ModelVarNames = input_vars,
        NonLinearities = create_dummy_nonlinearity(),
        ExprList = exprs,
        theta = theta,
        OutputVarName = "y",
        dtype = dtype,
        device = device,
    )


def compute_truth(
    model: SymbolicOscillator,
    y_lags: List[ int ],
    x_lags: List[ int ],
    theta: torch.Tensor,
    x: torch.Tensor,
    ds_data: Optional[ torch.Tensor ] = None,
    past_y: Optional[ torch.Tensor ] = None,
    past_x: Optional[ torch.Tensor ] = None,
) -> torch.Tensor:
  '''
  Manually compute the system output sequence y[k] for k=0..len(x)-1
  using the same expression structure as the model.
  `past_y` should contain y[-1], y[-2], ... (most recent first)
  `past_x` should contain x[-1], x[-2], ... (most recent first)
  '''
  n = x.shape[ 0 ]
  theta_dev = theta.to( x.device )
  y = torch.zeros( n, dtype = x.dtype, device = x.device )
  # Helper to fetch past y
  def get_y( k ):
    if ( k >= 0 ): return y[ k ] # R[1/3]
    else:
      idx = -k - 1 # k=-1 -> idx 0
      if ( ( past_y is None ) or ( idx >= len( past_y ) ) ):
        return torch.tensor( 0.0, dtype = y.dtype, device = y.device ) # R[2/3]
      return past_y[ idx ] # R[3/3]
  # Helper to fetch past x
  def get_x( k ):
    if ( k >= 0 ): return x[ k ] # R[1/3]
    else:
      idx = -k - 1
      if ( ( past_x is None ) or ( idx >= len( past_x ) ) ):
        return torch.tensor( 0.0, dtype = x.dtype, device = x.device ) # R[2/3]
      return past_x[ idx ] # R[3/3]

  for k in range( n ):
    val = torch.tensor( 0.0, dtype = y.dtype, device = y.device )
    idx = 0
    for lag in y_lags:
      val += theta_dev[ idx ] * get_y( k - lag )
      idx += 1
    for lag in x_lags:
      val += theta_dev[ idx ] * get_x( k - lag )
      idx += 1
    if ( ds_data is not None ): val += ds_data[ k ]
    y[ k ] = val
  return y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_model() -> tuple:
  '''Model with qy=1, qx=1.'''
  theta = torch.tensor( [ 0.5, 1.2, -0.7 ], dtype = torch.float64 )
  y_lags = [ 1 ]
  x_lags = [ 0, 1 ]
  model = create_model( y_lags, x_lags, theta )
  return model, y_lags, x_lags, theta


# ---------------------------------------------------------------------------
# Tests – Correctness
# ---------------------------------------------------------------------------
class TestBasicCorrectness:
  def test_simple_case( self, simple_model ) -> None:
    '''Basic correctness with qy=1, qx=1.'''
    model, y_lags, x_lags, theta = simple_model
    n = 20
    # Generate random input
    torch.manual_seed( 42 )
    x = torch.randn( n, dtype = torch.float64 )
    # True output (zero initial past)
    y = compute_truth( model, y_lags, x_lags, theta, x, ds_data = None,
                          past_y = torch.zeros( 10 ), past_x = torch.zeros( 10 ) )

    # Run helper
    y_hat = InitAndComputeBuffer( model, y, [ x ] )

    assert torch.allclose( y_hat, y, atol = 1e-12 )
    # Verify output properties match y
    assert y_hat.device == y.device
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert isinstance( y_hat, torch.Tensor )

  def test_with_dsdata( self ) -> None:
    '''DsData is injected additively.'''
    theta = torch.tensor( [ 0.3, 0.9 ], dtype = torch.float64 ) # only y[k-1] and x[k]
    y_lags = [ 1 ]
    x_lags = [ 0 ]
    model = create_model( y_lags, x_lags, theta )
    n = 15
    x = torch.randn( n, dtype = torch.float64 )
    ds = torch.randn( n, dtype = torch.float64 )

    y = compute_truth( model, y_lags, x_lags, theta, x, ds_data = ds,
                          past_y = torch.zeros( 5 ) )
    y_hat = InitAndComputeBuffer( model, y, [ x ], DsData = ds )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    assert isinstance( y_hat, torch.Tensor )

  def test_qx_different_from_qy( self ) -> None:
    '''Largest lags: qy=3, qx=2 -> StartIdx=3.'''
    theta = torch.tensor( [ 0.2, 0.05, 1.0, -0.5 ], dtype = torch.float64 )
    y_lags = [ 1, 3 ] # max qy = 3
    x_lags = [ 0, 2 ] # max qx = 2
    model = create_model( y_lags, x_lags, theta )
    n = 25
    x = torch.randn( n, dtype = torch.float64 )
    # Provide initial past values for negative indices
    past_y = torch.tensor( [ 0.1, -0.2, 0.05 ], dtype = torch.float64 ) # y[-1], y[-2], y[-3]
    past_x = torch.tensor( [ 0.0, 0.0 ], dtype = torch.float64 ) # x[-1], x[-2]
    y = compute_truth( model, y_lags, x_lags, theta, x,
                          past_y = past_y, past_x = past_x )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device

    # Verify internal storage is correctly initialised
    qx = model.get_MaxInputLag()
    qy = model.get_MaxOutputLag()
    start = max( qx, qy )
    # After Oscillate, output storage holds the last qy computed outputs
    stored_out = model.get_OutputStorage()
    assert stored_out.shape == ( qy, )
    assert stored_out.dtype == y.dtype
    assert torch.equal( stored_out.to( y.device ), y_hat[ -qy : ] )
    # Input storage: each row is the corresponding input slice
    stored_in = model.get_InputStorage()
    assert stored_in.shape[ 0 ] == 1 # one input variable
    assert stored_in.shape[ 1 ] == qx
    assert stored_in.dtype == x.dtype
    # After Oscillate, input storage holds the last qx values of the sliced data
    assert torch.equal( stored_in[ 0 ].to( x.device ), x[ start : ][ -qx : ] )

  def test_no_input_lag( self ) -> None:
    '''Purely AR system (qy>0, qx=0).'''
    theta = torch.tensor( [ 0.7, 0.2 ], dtype = torch.float64 ) # y[k-1], y[k-2]
    y_lags = [ 1, 2 ]
    x_lags = []
    model = create_model( y_lags, x_lags, theta )
    n = 30
    x_dummy = torch.zeros( n, dtype = torch.float64 ) # must still pass a Data tensor
    past_y = torch.tensor( [ 0.5, -0.3 ], dtype = torch.float64 )
    y = compute_truth( model, y_lags, x_lags, theta, x_dummy,
                          past_y = past_y )
    y_hat = InitAndComputeBuffer( model, y, [ x_dummy ] )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    # AR-only: output storage should hold last qy values; input storage empty
    stored_out = model.get_OutputStorage()
    assert stored_out.shape == ( max( y_lags ), )
    assert stored_out.dtype == y.dtype
    stored_in = model.get_InputStorage()
    assert stored_in.shape == ( 1, 0 )
    assert stored_in.dtype == x_dummy.dtype

  def test_no_output_lag( self ) -> None:
    '''Purely MA system (qy=0, qx>0).'''
    theta = torch.tensor( [ 1.0, -0.5 ], dtype = torch.float64 ) # x[k], x[k-1]
    y_lags = []
    x_lags = [ 0, 1 ]
    model = create_model( y_lags, x_lags, theta )
    n = 20
    x = torch.randn( n, dtype = torch.float64 )
    past_x = torch.tensor( [ 0.1 ], dtype = torch.float64 ) # x[-1]
    y = compute_truth( model, y_lags, x_lags, theta, x,
                          past_x = past_x )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    # MA-only: output storage empty, input storage holds last qx values
    stored_out = model.get_OutputStorage()
    assert stored_out.numel() == 0
    stored_in = model.get_InputStorage()
    assert stored_in.shape == ( 1, max( x_lags ) )
    assert stored_in.dtype == x.dtype
    assert torch.equal( stored_in[ 0 ].to( x.device ), x[ -max( x_lags ) : ] )

  def test_both_lags_zero( self ) -> None:
    '''Static system: y[k] = theta*x[k] (no lags).'''
    theta = torch.tensor( [ 2.5 ], dtype = torch.float64 )
    y_lags = []
    x_lags = [ 0 ]
    model = create_model( y_lags, x_lags, theta )
    n = 10
    x = torch.randn( n, dtype = torch.float64 )
    y = theta[ 0 ] * x # manual truth
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    assert isinstance( y_hat, torch.Tensor )
    # Static system: both storages empty
    stored_out = model.get_OutputStorage()
    assert stored_out.numel() == 0
    stored_in = model.get_InputStorage()
    assert stored_in.shape == ( 1, 0 )

  def test_original_data_unchanged( self, simple_model ) -> None:
    '''Original y and x are not mutated.'''
    model, y_lags, x_lags, theta = simple_model
    n = 12
    x = torch.randn( n, dtype = torch.float64 )
    y = compute_truth( model, y_lags, x_lags, theta, x )
    y_orig = y.clone()
    x_orig = x.clone()
    _ = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.equal( y, y_orig )
    assert torch.equal( x, x_orig )


# ---------------------------------------------------------------------------
# Tests – Error Handling
# ---------------------------------------------------------------------------
class TestInputValidation:
  def test_wrong_number_of_input_vars( self, simple_model ) -> None:
    '''Wrong number of input tensors raises ValueError.'''
    model, _, _, _ = simple_model
    n = 10
    x = torch.randn( n )
    y = torch.zeros( n )
    with pytest.raises( ValueError, match = "Number of input tensors" ) as exc:
      InitAndComputeBuffer( model, y, [ x, x ] ) # two instead of one
    assert "2" in str( exc.value ) and "1" in str( exc.value )

  def test_data_not_1d( self, simple_model ) -> None:
    '''2D input data raises ValueError.'''
    model, _, _, _ = simple_model
    x = torch.randn( 10, 2 ) # 2D
    y = torch.zeros( 10 )
    with pytest.raises( ValueError, match = "must be 1‑dimensional" ) as exc:
      InitAndComputeBuffer( model, y, [ x ] )
    assert "Data[0]" in str( exc.value )

  def test_dsdata_not_1d( self, simple_model ) -> None:
    '''2D DsData raises ValueError.'''
    model, _, _, _ = simple_model
    n = 10
    x = torch.randn( n )
    y = torch.zeros( n )
    ds = torch.randn( n, 1 ) # 2D
    with pytest.raises( ValueError, match = "DsData must be 1‑dimensional" ):
      InitAndComputeBuffer( model, y, [ x ], DsData = ds )

  def test_length_mismatch_y_vs_data( self, simple_model ) -> None:
    '''Length mismatch between y and Data raises ValueError.'''
    model, _, _, _ = simple_model
    x = torch.randn( 10 )
    y = torch.zeros( 12 ) # longer
    with pytest.raises( ValueError, match = "differs from  output length" ) as exc:
      InitAndComputeBuffer( model, y, [ x ] )
    assert "10" in str( exc.value ) and "12" in str( exc.value )

  def test_dsdata_length_mismatch( self, simple_model ) -> None:
    '''DsData length mismatch raises ValueError.'''
    model, _, _, _ = simple_model
    n = 10
    x = torch.randn( n )
    y = torch.zeros( n )
    ds = torch.randn( n + 1 )
    with pytest.raises( ValueError, match = "same length as y and Data" ):
      InitAndComputeBuffer( model, y, [ x ], DsData = ds )

  def test_sequence_too_short_for_lags( self ) -> None:
    '''Sequence shorter than StartIdx raises ValueError.'''
    theta = torch.tensor( [ 0.5, 0.1 ], dtype = torch.float64 )
    y_lags = [ 1 ]
    x_lags = [ 0 ]
    model = create_model( y_lags, x_lags, theta )
    n = 2 # StartIdx = 1 -> one sample remains for Oscillate
    x = torch.randn( n )
    y = torch.zeros( n )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    # The first sample is copied from y, the rest is computed by the model
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    # When sequence is shorter than StartIdx, the function raises
    with pytest.raises( ValueError, match = "Sequence too short to cover the required lags" ):
      InitAndComputeBuffer( model, torch.zeros( 0 ), [ torch.zeros( 0 ) ] )

  def test_positive_input_lag_raises( self ) -> None:
    '''Construct a model with a positive input lag (non-causal) and verify RuntimeError.'''
    theta = torch.tensor( [ 0.5, 0.2 ], dtype = torch.float64 )
    # Expressions: y[k-1] and x[k+1] (positive lag)
    exprs = [ "y[k-1]", "x[k+1]" ]
    model = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = create_dummy_nonlinearity(),
            ExprList = exprs,
            theta = theta,
            OutputVarName = "y",
        )
    n = 10
    x = torch.randn( n )
    y = torch.zeros( n )
    with pytest.raises( RuntimeError, match = "does not currently support models with positive input lags" ):
      InitAndComputeBuffer( model, y, [ x ] )


# ---------------------------------------------------------------------------
# Tests – Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
  def test_model_on_different_device_than_y( self, simple_model ) -> None:
    '''Ensure the output is moved to y.device even if model's device differs.'''
    model, y_lags, x_lags, theta = simple_model
    n = 10
    x = torch.randn( n, dtype = torch.float64 )
    y = compute_truth( model, y_lags, x_lags, theta, x )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert y_hat.device == y.device
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert torch.allclose( y_hat, y, atol = 1e-12 )

  def test_dsdata_none_explicit( self, simple_model ) -> None:
    '''Explicit None DsData works.'''
    model, y_lags, x_lags, theta = simple_model
    n = 15
    x = torch.randn( n, dtype = torch.float64 )
    y = compute_truth( model, y_lags, x_lags, theta, x )
    y_hat = InitAndComputeBuffer( model, y, [ x ], DsData = None )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device

  def test_large_lags_but_short_sequence( self ) -> None:
    '''qy=5, qx=4, sequence length just enough (StartIdx=5, n=6).'''
    theta = torch.ones( 9, dtype = torch.float64 ) # 5 y-lags + 4 x-lags
    y_lags = [ 1, 2, 3, 4, 5 ]
    x_lags = [ 0, 1, 2, 3 ]
    model = create_model( y_lags, x_lags, theta )
    n = 10 # StartIdx=5, remaining=5 >= MaxStartLag=5
    x = torch.randn( n, dtype = torch.float64 )
    past_y = torch.randn( 5, dtype = torch.float64 )
    past_x = torch.randn( 3, dtype = torch.float64 )
    y = compute_truth( model, y_lags, x_lags, theta, x,
                          past_y = past_y, past_x = past_x )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.allclose( y_hat, y, atol = 1e-12 )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    # Verify internal storage shapes reflect the large lags
    qy = model.get_MaxOutputLag()
    qx = model.get_MaxInputLag()
    stored_out = model.get_OutputStorage()
    assert stored_out.shape == ( qy, )
    stored_in = model.get_InputStorage()
    assert stored_in.shape == ( 1, qx )

  def test_nondefault_dtype( self ) -> None:
    '''Use float32.'''
    theta = torch.tensor( [ 0.5, 1.0 ], dtype = torch.float32 )
    y_lags = [ 1 ]
    x_lags = [ 0 ]
    model = create_model( y_lags, x_lags, theta, dtype = torch.float32 )
    n = 12
    x = torch.randn( n, dtype = torch.float32 )
    y = compute_truth( model, y_lags, x_lags, theta, x )
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert y_hat.dtype == torch.float32
    assert y_hat.device == y.device
    assert y_hat.shape == y.shape
    assert torch.allclose( y_hat, y, atol = 1e-6 )

  def test_output_storage_after_empty_qy( self ) -> None:
    '''When qy=0, OutputStorage shape is (0,). Verify no error.'''
    theta = torch.tensor( [ 1.0 ], dtype = torch.float64 )
    model = create_model( [], [ 0 ], theta ) # qy=0, qx=0
    n = 5
    x = torch.randn( n, dtype = torch.float64 )
    y = theta[ 0 ] * x
    y_hat = InitAndComputeBuffer( model, y, [ x ] )
    assert torch.allclose( y_hat, y )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    # get_OutputStorage should return an empty tensor
    out_stor = model.get_OutputStorage()
    assert out_stor.numel() == 0
    assert out_stor.dtype == y.dtype

  def test_input_storage_after_empty_qx( self ) -> None:
    '''When qx=0, InputStorage shape is (nInputVars, 0).'''
    theta = torch.tensor( [ 0.7, 0.3 ], dtype = torch.float64 )
    model = create_model( [ 1, 2 ], [], theta ) # qx=0
    n = 8
    x_dummy = torch.zeros( n, dtype = torch.float64 )
    past_y = torch.tensor( [ 0.2, -0.1 ], dtype = torch.float64 )
    y = compute_truth( model, [ 1, 2 ], [], theta, x_dummy, past_y = past_y )
    y_hat = InitAndComputeBuffer( model, y, [ x_dummy ] )
    assert torch.allclose( y_hat, y )
    assert y_hat.shape == y.shape
    assert y_hat.dtype == y.dtype
    assert y_hat.device == y.device
    in_stor = model.get_InputStorage()
    assert in_stor.shape == ( 1, 0 )
    assert in_stor.dtype == x_dummy.dtype
