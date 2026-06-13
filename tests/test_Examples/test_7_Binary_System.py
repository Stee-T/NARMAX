import os, dill
import torch as tor
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import NARMAX


def Bool_MAE( theta, L, ERR, RegNames, ValDic, DcFilterIdx = None ):
  if ( not isinstance( ValDic, dict ) ): raise AssertionError( "ValDic is not a dict" )
  for var in [ "y", "Data", "InputVarNames", "Lags" ]:
    if ( var not in ValDic ): raise AssertionError( f"'{ var }' missing" )
  if ( not isinstance( ValDic[ "Data" ], list ) ): raise AssertionError( "Data must be a list" )

  Error = 0.0
  for i in range( len( ValDic[ "Data" ] ) ):
    _, RegMat, RegNames = NARMAX.CTors.Lagger(
              ValDic[ "Data" ][ i ], ValDic[ "Lags" ], ValDic[ "InputVarNames" ]
          )
    RegMat, _ = NARMAX.CTors.Booler( RegMat, RegNames )
    if ( DcFilterIdx is not None ): RegMat = RegMat[ :, DcFilterIdx ]
    yHat = RegMat[ :, L.astype( np.int64 ) ].to( tor.get_default_dtype() ) @ theta
    Error += (
              tor.mean(
                  tor.abs( ValDic[ "y" ][ i ] - yHat )
                  / tor.mean( tor.abs( ValDic[ "y" ][ i ] ) )
              ).item()
          )
  return Error / len( ValDic[ "Data" ] )


def test_binary_miso_system() -> None:
  data_path = os.path.join( os.path.dirname( __file__ ), "test_7_Binary_System_data.pkl" )
  with open( data_path, "rb" ) as f:
    data = dill.load( f )

  y = data[ "y" ]
  RegMat = data[ "RegMat" ]
  RegNames = data[ "RegNames" ]
  ValidationDict = data[ "ValidationDict" ]
  expected_theta = data[ "expected_theta" ]
  expected_L = data[ "expected_L" ]
  expected_names = data[ "expected_names" ]

  tol = 0.0001
  ArboDepth = 3

  Arbo = NARMAX.Arborescence(
        y.to( tor.get_default_dtype() ),
        Dc = RegMat.to( tor.get_default_dtype() ),
        DcNames = RegNames,
        tolRoot = tol,
        tolRest = tol,
        MaxDepth = ArboDepth,
        ValFunc = Bool_MAE,
        ValData = ValidationDict,
    )

  theta, L, ERR, Morphdict, RegMat_out, RegNames_out = Arbo.fit()

  # ---- Type and shape checks on fit() output ----
  assert isinstance( theta, tor.Tensor ), f"theta must be a torch.Tensor, got {type(theta)}"
  assert isinstance( L, np.ndarray ), f"L must be an np.ndarray, got {type(L)}"
  assert isinstance( ERR, np.ndarray ), f"ERR must be an np.ndarray, got {type(ERR)}"
  assert isinstance( RegMat_out, tor.Tensor ), f"RegMat must be a torch.Tensor, got {type(RegMat_out)}"
  assert isinstance( RegNames_out, np.ndarray ), f"RegNames must be an np.ndarray, got {type(RegNames_out)}"
  assert theta.ndim == 1, f"theta must be 1D, got shape {theta.shape}"
  assert theta.dtype == tor.get_default_dtype(), f"theta dtype {theta.dtype} != default {tor.get_default_dtype()}"
  assert L.ndim == 1, f"L must be 1D, got shape {L.shape}"
  assert L.dtype == np.int64, f"L dtype {L.dtype} != int64"
  assert ERR.ndim == 1, f"ERR must be 1D, got shape {ERR.shape}"
  assert ERR.dtype == np.float64, f"ERR dtype {ERR.dtype} != float64"
  assert RegMat_out.ndim == 2, f"RegMat must be 2D, got shape {RegMat_out.shape}"
  assert RegNames_out.ndim == 1, f"RegNames must be 1D, got shape {RegNames_out.shape}"
  assert all( isinstance( n, str ) for n in RegNames_out ), "RegNames must contain only strings"

  # ---- Sanity checks ------------------------------
  assert len( L ) != 0, "Expected no dictionary candidates"
  assert len( theta ) != 0, "Expected non-empty theta"
  assert len( ERR ) != 0, "Expected non-empty ERR"
  assert len( RegMat_out ) != 0, "Expected non-empty RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert len( RegNames_out ) != 0, "Expected non-empty RegNames"

  # ---- Consistency checks on fit() results ----
  assert len( theta ) == len( L ) == len( ERR ), f"Length mismatch: theta {len(theta)}, L {len(L)}, ERR {len(ERR)}"
  assert len( L ) == len( np.unique( L ) ), "L must contain unique regressor indices"
  assert np.all( L >= 0 ), "L contains negative indices"
  assert np.all( L < RegMat_out.shape[ 1 ] ), f"L indices exceed RegMat columns ({RegMat_out.shape[1]})"
  assert tor.all( tor.isfinite( theta ) ), "theta contains non-finite values"
  assert tor.all( tor.abs( theta ) > 0 ), "theta contains zero coefficients"
  assert np.all( np.isfinite( ERR ) ), "ERR contains non-finite values"
  assert np.all( ERR >= 0 ), "ERR contains negative values"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx = np.argsort( L )
  L_sorted = np.asarray( L )[ sort_idx ]
  theta_sorted = theta[ sort_idx ].cpu().numpy()
  names_sorted = np.asarray( RegNames )[ L_sorted ]

  # ---- Check that sorted L is strictly increasing ----
  assert np.all( np.diff( L_sorted ) > 0 ), "Sorted L is not strictly increasing"

  # ---- Call Bool_MAE validation function and check its return type/value ----
  val_error = Bool_MAE( theta, L, ERR, RegNames, ValidationDict )
  assert isinstance( val_error, float ), f"Validation error must be float, got {type(val_error)}"
  assert val_error >= 0.0, f"Validation error must be non-negative, got {val_error}"
  assert np.isfinite( val_error ), f"Validation error must be finite, got {val_error}"

  # ---- 8. Assertions against recorded reference ---------------------------------------
  np.testing.assert_array_equal( L_sorted, expected_L )
  np.testing.assert_array_equal( names_sorted, expected_names )
  np.testing.assert_allclose( theta_sorted, expected_theta, rtol = 1e-5, atol = 1e-7 )
