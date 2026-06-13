import numpy as np
import torch as tor
import scipy.signal as sps
import matplotlib
matplotlib.use( "Agg" )
import pytest

import NARMAX


def Sys( x, Filters, Print = False ):
  if ( len( Filters ) < 1 ): raise ValueError( "There must be at least one filter" )
  if ( Print ):
    for i, ( b, a ) in enumerate( Filters ): print( f"Biquad { i + 1 }:\nb = { b }\na = { a }" )
    print()
  y = np.zeros_like( x.cpu().numpy() )
  for b, a in Filters: y += sps.lfilter( b, a, x.cpu().numpy() )
  return tor.tensor( y )


def test_iir_parallel_filters() -> None:
  '''Integration test for IIR parallel filter fitting.'''
  # Fixed seeds for reproducibility
  tor.manual_seed( 42 ) # for torch.rand in validation generation
  np.random.seed( 3 ) # global numpy seed, but we use a Generator for filters

  # ---------------------------------------------------------------------------------------
  p = 2_000
  Amplitude = 1.0
  ExpansionOrder = 1
  W = None
  Fs = 44_100

  nBiquads = 5
  qx = 2 * nBiquads
  qy = 2 * nBiquads

  # Random filter generation (deterministic from seed=3)
  RNG = np.random.default_rng( seed = 3 )

  def RandomPeakIIR( FreqRange, QRange, Fs ):
    f0 = RNG.integers( int( FreqRange[ 0 ] ), int( FreqRange[ 1 ] ) )
    Q = RNG.uniform( QRange[ 0 ], QRange[ 1 ] )
    b, a = sps.iirpeak( f0, Q, Fs )
    return b / a[ 0 ], a / a[ 0 ]

  Filters = [ RandomPeakIIR( [ 10, 8_000 ], [ 15, 30 ], Fs ) for _ in range( nBiquads ) ]

  # Generate training data
  x = tor.tensor( RNG.uniform( -Amplitude, Amplitude, size = p ) ) # uniform white noise
  x -= tor.mean( x )
  y = Sys( x, Filters )

  # Training data: only lagged regressors
  y, RegMat, RegNames = NARMAX.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) )

  # Validation data
  DsValDict = {
        "y": [],
        "Data": [],
        "InputVarNames": [ "x", "y" ],
        "NonLinearities": [ NARMAX.Identity ],
    }

  for i in range( 5 ):
    x_val = tor.rand( int( p / 3 ) )
    x_val -= tor.mean( x_val )
    y_val = Sys( x_val, Filters )
    DsValDict[ "y" ].append( y_val )
    DsValDict[ "Data" ].append( [ x_val ] )

  # Fit imposed regressors (no selection)
  Arbo = NARMAX.Arborescence(
        y,
        Ds = RegMat,
        DsNames = RegNames,
        ValFunc = NARMAX.DefaultValidation,
        ValData = DsValDict,
    )

  _ = Arbo.fit() # only need the coefficients
  theta, L, ERR, Morphdict, RegMat_out, _ = Arbo.get_Results()

  # ---- Sanity checks ------------------------------
  assert isinstance( Arbo, NARMAX.Arborescence )
  assert isinstance( L, ( list, np.ndarray ) ), "L must be a list or numpy array"
  assert len( L ) == 0, "Expected no dictionary candidates (Ds fitting only)"
  assert isinstance( theta, tor.Tensor ), "theta must be a torch tensor"
  assert theta.ndim == 1, "theta must be 1-D"
  assert len( theta ) == 21, f"Expected 21 coefficients, got {len(theta)}"
  assert isinstance( ERR, ( tor.Tensor, np.ndarray ) ), "ERR must be a torch tensor or numpy array"
  assert ERR.ndim == 1, "ERR must be 1-D"
  assert len( ERR ) == 21, f"Expected 21 ERR values, got {len(ERR)}"
  assert np.all( ERR >= 0 ), "ERR values must be non-negative"
  assert np.all( ERR <= 1 ), "ERR values must not exceed 1"
  assert RegMat_out is None, "Expected None RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert isinstance( RegNames, ( list, np.ndarray ) ), "RegNames must be a list or numpy array"
  assert len( RegNames ) == 21, f"Expected 21 regressor names, got {len(RegNames)}"
  assert all( isinstance( n, str ) for n in RegNames ), "All RegNames must be strings"

  np.testing.assert_allclose( theta.cpu().numpy(), np.array( [ 9.06432968e-02, -5.41765570e-01, 1.45626193e+00, -2.18491890e+00,
                                                  1.68211416e+00, 9.83792355e-02, -1.79073507e+00, 2.14169606e+00,
                                                  -1.35933793e+00, 4.86609180e-01, -7.89463840e-02, 7.26415880e+00,
                                                  -2.56636766e+01, 5.75198093e+01, -9.01224766e+01, 1.02900359e+02,
                                                  -8.66512387e+01, 5.31963103e+01, -2.28512415e+01, 6.23745792e+00,
                                                  -8.30774985e-01 ] ),
                                atol = 1e-3, rtol = 1e-3
                              )

  # Convert to IIR coefficients
  IIRResult = NARMAX.Tools.rFOrLSR2IIR( theta, [ i for i in range( len( theta ) ) ], RegNames )
  assert isinstance( IIRResult, tuple ), "rFOrLSR2IIR must return a tuple"
  assert len( IIRResult ) == 2, "rFOrLSR2IIR must return (b, a)"
  b_Ds, a_Ds = IIRResult
  assert isinstance( b_Ds, np.ndarray ), "b_Ds must be numpy array"
  assert isinstance( a_Ds, np.ndarray ), "a_Ds must be numpy array"
  assert b_Ds.ndim == 1, "b_Ds must be 1-D"
  assert a_Ds.ndim == 1, "a_Ds must be 1-D"
  assert a_Ds[ 0 ] == 1.0, "a_Ds[0] must be 1 (normalized)"

  # ---------------------------------------------------------------
  # Assertion: frequency response of estimated IIR matches original filter bank
  Resolution = 5_000
  w, h_estimated = sps.freqz( b_Ds, a_Ds, worN = Resolution, fs = Fs )

  assert len( w ) == Resolution, f"Expected {Resolution} frequency points, got {len(w)}"
  assert len( h_estimated ) == Resolution, f"Expected {Resolution} freqz points, got {len(h_estimated)}"

  # Original parallel filters frequency response
  h_original = np.sum( [ sps.freqz( filt[ 0 ], filt[ 1 ], worN = Resolution, fs = Fs )[ 1 ] for filt in Filters ], axis = 0, )

  # Compare complex responses within tight tolerance
  np.testing.assert_allclose( np.abs( h_estimated ), np.abs( h_original ), atol = 1e-2, rtol = 1e-2 )
  np.testing.assert_allclose( h_estimated, h_original, atol = 1e-2, rtol = 1e-2 )

def test_sys_rejects_empty_filters() -> None:
  '''Sys() must raise ValueError when Filters list is empty.'''
  x = tor.zeros( 10 )
  err_msg = "There must be at least one filter"

  with pytest.raises( ValueError, match = err_msg ):
    Sys( x, [] )


def test_validation_dict_keys() -> None:
  '''Check that DsValDict has all required keys with correct types.'''
  DsValDict = {
    "y": [ tor.randn( 10 ) ],
    "Data": [ [ tor.randn( 10 ) ] ],
    "InputVarNames": [ "x", "y" ],
    "NonLinearities": [ NARMAX.Identity ],
  }

  assert "y" in DsValDict
  assert "Data" in DsValDict
  assert "InputVarNames" in DsValDict
  assert "NonLinearities" in DsValDict
  assert isinstance( DsValDict[ "y" ], list )
  assert isinstance( DsValDict[ "Data" ], list )
  assert isinstance( DsValDict[ "InputVarNames" ], list )
  assert isinstance( DsValDict[ "NonLinearities" ], list )
  assert len( DsValDict[ "y" ] ) > 0
  assert len( DsValDict[ "Data" ] ) > 0
  assert len( DsValDict[ "InputVarNames" ] ) == 2
  assert len( DsValDict[ "NonLinearities" ] ) == 1
  for v in DsValDict[ "y" ]:
    assert isinstance( v, tor.Tensor )
  for v in DsValDict[ "Data" ]:
    assert isinstance( v, list )
    for sub in v:
      assert isinstance( sub, tor.Tensor )


if ( __name__ == "__main__" ): test_iir_parallel_filters()
