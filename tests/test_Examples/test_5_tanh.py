import pytest
import torch as tor
import numpy as np
import matplotlib
matplotlib.use( "Agg" ) # non‑interactive backend for headless environments
import matplotlib.pyplot as plt

import NARMAX


def Sigmoid_Expansion_L_inf( theta, L, ERR, RegNames, ValDic, DcFilterIdx = None ):
  if ( not isinstance( ValDic, dict ) ): raise AssertionError( "ValDic is not a dict" )
  for var in [ "y", "Data", "ExpansionOrder" ]:
    if ( var not in ValDic ): raise AssertionError( f"'{ var }' missing" )
  if ( not isinstance( ValDic[ "Data" ], list ) ): raise AssertionError( "Data must be a list" )

  Error = 0.0
  for i in range( len( ValDic[ "Data" ] ) ):
    RegMat, _ = NARMAX.CTors.Expander(
              Data = ValDic[ "Data" ][ i ],
              RegNames = [ "|x|" ],
              ExpansionOrder = ValDic[ "ExpansionOrder" ]
          )
    if ( DcFilterIdx is not None ): RegMat = RegMat[ :, DcFilterIdx ]
    A = tor.abs( RegMat[ :, L.astype( np.int64 ) ] ) @ theta
    yHat = tor.sign( ValDic[ "Data" ][ i ] ).view( -1 ) * ( 1 - 1 / ( 1 + A ) )
    Error += tor.max( tor.abs( ValDic[ "y" ][ i ] - yHat ) )
  return Error / len( ValDic[ "Data" ] )


def test_validation_function_error_messages() -> None:
  theta_dummy = tor.tensor( [ 1.0 ] )
  L_dummy = np.array( [ 0 ], dtype = np.int64 )
  ERR_dummy = np.array( [ 1.0 ] )
  RegNames_dummy = np.array( [ "|x|" ] )
  ValDic_valid = { "y": [ tor.randn( 10 ) ], "Data": [ tor.randn( 10, 1 ) ], "ExpansionOrder": 5 }

  with pytest.raises( AssertionError, match = "ValDic is not a dict" ):
    Sigmoid_Expansion_L_inf( theta_dummy, L_dummy, ERR_dummy, RegNames_dummy, ValDic = "not_a_dict" )

  for missing_key in [ "y", "Data", "ExpansionOrder" ]:
    ValDic_missing = { k: v for k, v in ValDic_valid.items() if k != missing_key }
    with pytest.raises( AssertionError, match = f"'{ missing_key }' missing" ):
      Sigmoid_Expansion_L_inf( theta_dummy, L_dummy, ERR_dummy, RegNames_dummy, ValDic = ValDic_missing )

  ValDic_bad_data = { "y": [ tor.randn( 10 ) ], "Data": tor.randn( 10, 1 ), "ExpansionOrder": 5 }
  with pytest.raises( AssertionError, match = "Data must be a list" ):
    Sigmoid_Expansion_L_inf( theta_dummy, L_dummy, ERR_dummy, RegNames_dummy, ValDic = ValDic_bad_data )


def test_validation_function_with_dcfilter() -> None:
  L_dummy = np.array( [ 0 ], dtype = np.int64 )
  theta_dummy = tor.tensor( [ 1.0 ] )
  ERR_dummy = np.array( [ 1.0 ] )
  RegNames_dummy = np.array( [ "|x|" ] )
  x_val = tor.linspace( -8.0, 8.0, 1_000 )
  ValDic = {
    "y": [ tor.tanh( x_val ) ],
    "Data": [ x_val.view( -1, 1 ) ],
    "ExpansionOrder": 5,
  }
  err = Sigmoid_Expansion_L_inf( theta_dummy, L_dummy, ERR_dummy, RegNames_dummy, ValDic, DcFilterIdx = np.array( [ 0 ] ) )
  assert isinstance( err, ( float, tor.Tensor, np.floating ) ), "Error should be numeric"
  assert err >= 0.0, "Error must be non-negative"
  err_none = Sigmoid_Expansion_L_inf( theta_dummy, L_dummy, ERR_dummy, RegNames_dummy, ValDic, DcFilterIdx = None )
  assert err_none >= 0.0, "Error must be non-negative"


def test_sigmoid_expansion() -> None:
  '''Integration test for sigmoid (tanh) expansion using Arborescence.'''
  # Reproducibility
  tor.manual_seed( 0 )
  np.random.seed( 0 )

  # Inputs creation
  p = 50_000
  TestRange = 8.0
  ExpansionOrder = 13
  Amplitude = 2.45
  tol = 1e-9

  x = ( 2 * Amplitude ) * ( tor.rand( p ) - 0.5 )
  x -= tor.mean( x )
  y = tor.tanh( x )

  # Training data
  RegMat, RegNames = NARMAX.CTors.Expander(
        Data = tor.abs( x.view( -1, 1 ) ),
        RegNames = [ "|x|" ],
        ExpansionOrder = ExpansionOrder
    )
  y = -y / tor.sign( x ) + 1
  RegMat = -y.view( -1, 1 ) * RegMat

  # Validation data
  xValidation = tor.linspace( -TestRange, TestRange, 50_000 )
  yValidation = tor.tanh( xValidation )

  ValidationDict = {
        "y": [ yValidation ],
        "Data": [ xValidation.view( -1, 1 ) ],
        "ExpansionOrder": ExpansionOrder,
    }

  # Fit
  Arbo = NARMAX.Arborescence(
        y,
        Dc = RegMat,
        DcNames = RegNames,
        tolRoot = tol,
        tolRest = tol,
        MaxDepth = 3,
        ValFunc = Sigmoid_Expansion_L_inf,
        ValData = ValidationDict,
    )
  theta, L, ERR, Morphdict, RegMat, RegNames = Arbo.fit()

  # ---- Sanity checks ------------------------------
  assert len( L ) != 0, "Expected no dictionary candidates"
  assert len( theta ) != 0, "Expected non-empty theta"
  assert len( ERR ) != 0, "Expected non-empty ERR"
  assert len( RegMat ) != 0, "Expected non-empty RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert len( RegNames ) != 0, "Expected non-empty RegNames"

  # ---- Type checks ------------------------------
  assert isinstance( theta, tor.Tensor ), "theta should be a torch.Tensor"
  assert isinstance( L, np.ndarray ), "L should be a numpy array"
  assert isinstance( ERR, np.ndarray ), "ERR should be a numpy array"
  assert isinstance( RegMat, tor.Tensor ), "RegMat should be a torch.Tensor"
  assert isinstance( RegNames, np.ndarray ), "RegNames should be a numpy array"

  assert theta.dtype.is_floating_point, "theta should be a float tensor"
  assert L.dtype == np.int64, "L should be int64"
  assert ERR.dtype == np.float64, "ERR should be float64"

  # ---- Shape consistency checks ------------------------------
  n_selected = len( L )
  assert theta.shape == ( n_selected, ), f"theta shape { theta.shape } != ({ n_selected },)"
  assert ERR.shape == ( n_selected, ), f"ERR shape { ERR.shape } != ({ n_selected },)"
  assert RegMat.shape[ 0 ] == p, f"RegMat rows { RegMat.shape[ 0 ] } != { p }"
  assert RegMat.shape[ 1 ] == len( RegNames ), f"RegMat cols { RegMat.shape[ 1 ] } != len(RegNames) { len( RegNames ) }"

  # ---- ERR checks ------------------------------
  assert np.all( ERR >= 0 ), "All ERR values should be non-negative"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx = np.argsort( L )
  L_sorted = np.asarray( L )[ sort_idx ]
  theta_sorted = theta[ sort_idx ].cpu().numpy()
  names_sorted = np.asarray( RegNames )[ L_sorted ] # select & order names

  # ---- 7. Expected values (known example results) ---------------------------------------
  expected_theta = np.array( [ 1.00107063e+00, 9.90735703e-01, 7.00563738e-01, 2.70358426e-01,
                                             2.00667156e-01, 3.06767045e-02, 5.43299284e-04 ] )

  expected_L = np.array( [ 0, 1, 2, 3, 4, 6, 9 ] )
  expected_names = np.array( [ '|x|', '|x|^2', '|x|^3', '|x|^4', '|x|^5', '|x|^7', '|x|^10' ], dtype = '<U6' )

  # ---- 8. Assertions -------------------------------------------------------
  np.testing.assert_array_equal( L_sorted, expected_L )
  np.testing.assert_array_equal( names_sorted, expected_names )
  np.testing.assert_allclose( theta_sorted, expected_theta, rtol = 1e-3, atol = 1e-3 )


  # ---- 9. Extra checks: model must reproduce tanh accurately ----
  x_test = tor.linspace( -TestRange, TestRange, 10_000 )
  RegMat_test, _ = NARMAX.CTors.Expander(
        Data = x_test.view( -1, 1 ),
        RegNames = [ "x" ],
        ExpansionOrder = ExpansionOrder
    )
  A_test = tor.abs( RegMat_test[ :, L.astype( np.int64 ) ] ) @ theta
  yPred = tor.sign( x_test ) * ( 1 - 1 / ( 1 + A_test ) )
  yTrue = tor.tanh( x_test )

  median_err = tor.median( tor.abs( yTrue - yPred ) ).item()
  assert median_err < 1e-5, f"Median error { median_err } exceeds tolerance"

  max_err = tor.max( tor.abs( yTrue - yPred ) ).item()
  assert max_err < 1e-3, f"Max error { max_err } exceeds tolerance"

  # Check model output at x = 0  (tanh(0) = 0)
  x0 = tor.tensor( [ 0.0 ] )
  RegMat0, _ = NARMAX.CTors.Expander(
        Data = x0.view( -1, 1 ),
        RegNames = [ "x" ],
        ExpansionOrder = ExpansionOrder
    )
  A0 = tor.abs( RegMat0[ :, L.astype( np.int64 ) ] ) @ theta
  yPred0 = tor.sign( x0 ) * ( 1 - 1 / ( 1 + A0 ) )
  assert abs( yPred0.item() ) < 1e-5, f"Model at x=0 should be near 0, got { yPred0.item() }"

  # Check anti-symmetry: f(-x) should equal -f(x)
  x_pair = tor.tensor( [ [ 2.5 ], [ -2.5 ] ] )
  RegMat_pair, _ = NARMAX.CTors.Expander(
        Data = x_pair,
        RegNames = [ "x" ],
        ExpansionOrder = ExpansionOrder
    )
  A_pair = tor.abs( RegMat_pair[ :, L.astype( np.int64 ) ] ) @ theta
  yPair = tor.sign( x_pair ).squeeze( -1 ) * ( 1 - 1 / ( 1 + A_pair ) )
  assert abs( yPair[ 0 ].item() + yPair[ 1 ].item() ) < 1e-5, "Model should be anti-symmetric"
