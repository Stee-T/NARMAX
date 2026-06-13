import torch as tor
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import NARMAX
import NARMAX.TestSystems as Test_Systems


def test_binary_miso_system() -> None:
  '''Integration test for a binary MISO system.'''
  # Reproducibility
  tor.manual_seed( 1243 )
  np.random.seed( 1243 )

  # ---------------------------------------------------- 2. Hyper-parameters
  p = 2_000
  tol = 0.0001
  W = None
  ArboDepth = 3

  # ---------------------------------------------------- 3. Training Data
  while ( True ):
    x1 = tor.randint( 0, 2, ( p, ), dtype = tor.bool )
    x2 = tor.randint( 0, 2, ( p, ), dtype = tor.bool )
    x3 = tor.randint( 0, 2, ( p, ), dtype = tor.bool )
    x4 = tor.randint( 0, 2, ( p, ), dtype = tor.bool )

    x1, x2, x3, x4, y, W = Test_Systems.Binary_MISO_System(
            x1, x2, x3, x4, W, Print = False
        )
    if ( not tor.isnan( tor.sum( y ) ) ): break

  # ---- Type/shape checks on generated training data ----
  assert x1.dtype == tor.bool and x2.dtype == tor.bool and x3.dtype == tor.bool and x4.dtype == tor.bool
  assert y.dtype == tor.int32, f"y dtype should be int32, got {y.dtype}"
  assert W is None, "W should remain None when passed as None"
  assert y.ndim == 1, f"y should be 1D, got shape {y.shape}"
  assert len( y ) == len( x1 ), f"y length {len(y)} != x1 length {len(x1)}"
  assert not tor.isnan( tor.sum( y ) ), "y still contains NaN after loop"
  assert tor.all( ( y >= -8 ) & ( y <= 8 ) ), f"y values outside expected [-8, 8] range"

  VarNames = [ "x1", "x2", "x3", "x4" ]
  Lags = [ 2, 2, 2, 2 ]
  y, RegMat, RegNames = NARMAX.CTors.Lagger(
        [ x1, x2, x3, x4, y ], Lags + [ 0 ], VarNames + [ "y" ]
    )

  # ---- Type/shape checks after Lagger ----
  assert isinstance( y, tor.Tensor ), "y must be a torch.Tensor after Lagger"
  assert isinstance( RegMat, tor.Tensor ), "RegMat must be a torch.Tensor after Lagger"
  assert isinstance( RegNames, np.ndarray ), "RegNames must be an np.ndarray after Lagger"
  assert y.ndim == 1, f"y must be 1D after Lagger, got shape {y.shape}"
  assert RegMat.ndim == 2, f"RegMat must be 2D after Lagger, got shape {RegMat.shape}"
  assert RegMat.shape[ 0 ] == len( y ), f"RegMat rows {RegMat.shape[0]} != y length {len(y)}"
  assert RegMat.shape[ 1 ] == len( RegNames ), f"RegMat cols {RegMat.shape[1]} != RegNames length {len(RegNames)}"
  assert all( isinstance( n, str ) for n in RegNames ), "RegNames must contain only strings"

  RegMat, RegNames = NARMAX.CTors.Booler( RegMat, RegNames )

  # ---- Type/shape checks after Booler ----
  assert RegMat.dtype == tor.bool, f"Booler RegMat dtype should be bool, got {RegMat.dtype}"
  assert isinstance( RegNames, np.ndarray ), "RegNames must be an np.ndarray after Booler"
  assert RegMat.ndim == 2, f"RegMat must be 2D after Booler, got shape {RegMat.shape}"
  assert RegMat.shape[ 0 ] == y.shape[ 0 ], f"Booler changed row count: {RegMat.shape[0]} vs {y.shape[0]}"
  # Verify no constant columns (all True or all False)
  col_sums = RegMat.sum( dim=0 )
  assert tor.all( col_sums > 0 ), "Booler produced constant-False columns"
  assert tor.all( col_sums < RegMat.shape[ 0 ] ), "Booler produced constant-True columns"

  # ---------------------------------------------------- 4. Validation Data
  ValidationDict = {
        "y": [],
        "Data": [],
        "InputVarNames": VarNames,
        "Lags": Lags,
    }

  for i in range( 5 ):
    x1_val = tor.randint( 0, 2, ( p // 2, ), dtype = tor.bool )
    x2_val = tor.randint( 0, 2, ( p // 2, ), dtype = tor.bool )
    x3_val = tor.randint( 0, 2, ( p // 2, ), dtype = tor.bool )
    x4_val = tor.randint( 0, 2, ( p // 2, ), dtype = tor.bool )

    _, _, _, _, y_val, W = Test_Systems.Binary_MISO_System(
            x1_val, x2_val, x3_val, x4_val, W, Print = False
        )

    ValidationDict[ "y" ].append( y_val.to( tor.float64 ) )
    ValidationDict[ "Data" ].append( [
                                        x1_val.to( tor.float64 ),
                                        x2_val.to( tor.float64 ),
                                        x3_val.to( tor.float64 ),
                                        x4_val.to( tor.float64 ),
                                      ] )

  # ---------------------------------------------------- 4. Validation function
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

  # ---------------------------------------------------- 5. Running the Arborescence
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

  theta, L, ERR, Morphdict, RegMat, RegNames = Arbo.fit()

  # ---- Type and shape checks on fit() output ----
  assert isinstance( theta, tor.Tensor ), f"theta must be a torch.Tensor, got {type(theta)}"
  assert isinstance( L, np.ndarray ), f"L must be an np.ndarray, got {type(L)}"
  assert isinstance( ERR, np.ndarray ), f"ERR must be an np.ndarray, got {type(ERR)}"
  assert isinstance( RegMat, tor.Tensor ), f"RegMat must be a torch.Tensor, got {type(RegMat)}"
  assert isinstance( RegNames, np.ndarray ), f"RegNames must be an np.ndarray, got {type(RegNames)}"
  assert theta.ndim == 1, f"theta must be 1D, got shape {theta.shape}"
  assert theta.dtype == tor.get_default_dtype(), f"theta dtype {theta.dtype} != default {tor.get_default_dtype()}"
  assert L.ndim == 1, f"L must be 1D, got shape {L.shape}"
  assert L.dtype == np.int64, f"L dtype {L.dtype} != int64"
  assert ERR.ndim == 1, f"ERR must be 1D, got shape {ERR.shape}"
  assert ERR.dtype == np.float64, f"ERR dtype {ERR.dtype} != float64"
  assert RegMat.ndim == 2, f"RegMat must be 2D, got shape {RegMat.shape}"
  assert RegNames.ndim == 1, f"RegNames must be 1D, got shape {RegNames.shape}"
  assert all( isinstance( n, str ) for n in RegNames ), "RegNames must contain only strings"

  # ---- Sanity checks ------------------------------
  assert len( L ) != 0, "Expected no dictionary candidates"
  assert len( theta ) != 0, "Expected non-empty theta"
  assert len( ERR ) != 0, "Expected non-empty ERR"
  assert len( RegMat ) != 0, "Expected non-empty RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert len( RegNames ) != 0, "Expected non-empty RegNames"

  # ---- Consistency checks on fit() results ----
  assert len( theta ) == len( L ) == len( ERR ), f"Length mismatch: theta {len(theta)}, L {len(L)}, ERR {len(ERR)}"
  assert len( L ) == len( np.unique( L ) ), "L must contain unique regressor indices"
  assert np.all( L >= 0 ), "L contains negative indices"
  assert np.all( L < RegMat.shape[ 1 ] ), f"L indices exceed RegMat columns ({RegMat.shape[1]})"
  assert tor.all( tor.isfinite( theta ) ), "theta contains non-finite values"
  assert tor.all( tor.abs( theta ) > 0 ), "theta contains zero coefficients"
  assert np.all( np.isfinite( ERR ) ), "ERR contains non-finite values"
  assert np.all( ERR >= 0 ), "ERR contains negative values"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx = np.argsort( L )
  L_sorted = np.asarray( L )[ sort_idx ]
  theta_sorted = theta[ sort_idx ].cpu().numpy()
  names_sorted = np.asarray( RegNames )[ L_sorted ] # select & order names

  # ---- Check that sorted L is strictly increasing ----
  assert np.all( np.diff( L_sorted ) > 0 ), "Sorted L is not strictly increasing"

  # ---- Call Bool_MAE validation function and check its return type/value ----
  val_error = Bool_MAE( theta, L, ERR, RegNames, ValidationDict )
  assert isinstance( val_error, float ), f"Validation error must be float, got {type(val_error)}"
  assert val_error >= 0.0, f"Validation error must be non-negative, got {val_error}"
  assert np.isfinite( val_error ), f"Validation error must be finite, got {val_error}"

  # ---- 7. Expected values (known example results) ---------------------------------------
  expected_theta = np.array( [ 1., 1., -1., 1., -1., 1., -1., 1. ] )
  expected_L = np.array( [ 10, 37, 63, 186, 193, 355, 392, 522 ] )
  expected_names = np.array( [ 'x4[k-1]', '!x1[k] && x2[k]', '!x1[k] && !x3[k-1]',
                                             '!x2[k-1] && x3[k-2]', 'x2[k-1] && !x3[k-2]', 'x2[k] ^ x4[k-2]',
                                             'x3[k] ^ x4[k]', 'x1[k-2] || !x2[k]' ], dtype = '<U19'
                                           )

  # ---- 8. Assertions -------------------------------------------------------
  np.testing.assert_array_equal( L_sorted, expected_L )
  np.testing.assert_array_equal( names_sorted, expected_names )
  np.testing.assert_allclose( theta_sorted, expected_theta, rtol = 1e-5, atol = 1e-7 )
