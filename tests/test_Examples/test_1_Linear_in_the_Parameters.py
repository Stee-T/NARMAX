import numpy as np
from numpy.typing import NDArray
import torch as tor
import pytest
import matplotlib
matplotlib.use( "Agg" ) # non-interactive backend – safe for CI

import NARMAX
import NARMAX.TestSystems as Test_Systems


def test_nonlinearities_arborescence() -> None:
  '''Integration test for the NonLinearities system with an Arborescence.'''

  # ---- Fixed seed (PyTorch) ------------------------------------------------
  seed = 42
  tor.manual_seed( seed )

  # ---- 1. Imports inside function (fine, but already global) ----------------
  # (done above)

  # ---- 2. Hyper-parameters --------------------------------------------------
  p: int = 2_500
  InputAmplitude: float = 1.0
  tol: float = 0.0001
  W = None
  qx: int = 4
  qy: int = 4
  ExpansionOrder: int = 3
  ArboDepth: int = 4

  Sys = Test_Systems.NonLinearities

  # ---- Generate training data -----------------------------------------------
  max_attempts = 100
  for _ in range( max_attempts ):
    x = ( InputAmplitude * 2 ) * ( tor.rand( p ) - 0.5 )
    x -= tor.mean( x )
    x, y, W = Sys( x, W, Print = False )
    if ( not tor.isnan( tor.sum( y ) ) ): break
  else: raise RuntimeError( f"Could not generate NaN-free data after { max_attempts } attempts." )

  NonLinearities = [ NARMAX.Identity ]
  NonLinearities.append( NARMAX.NonLinearity( "abs", f = tor.abs ) )
  NonLinearities.append( NARMAX.NonLinearity( "cos", f = tor.cos ) )
  NonLinearities.append( NARMAX.NonLinearity( "exp", f = tor.exp ) )

  # ---- 3. Training data ----------------------------------------------------
  y, RegMat, RegNames = NARMAX.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) )
  RegMat, RegNames = NARMAX.CTors.Expander( RegMat, RegNames, ExpansionOrder )
  RegMat, RegNames, _ = NARMAX.CTors.NonLinearizer( y, RegMat, RegNames, NonLinearities )

  # ---- 4. Validation data --------------------------------------------------
  ValidationDict = {
        "y": [],
        "Data": [],
        "InputVarNames": [ "x", "y" ],
        "NonLinearities": NonLinearities,
    }
  for i in range( 5 ):
    for _ in range( max_attempts ):
      x_val = tor.rand( int( p / 3 ) )
      x_val -= tor.mean( x_val )
      x_val, y_val, W = Sys( x_val, W, Print = False )
      if ( not tor.isnan( tor.sum( y_val ) ) ): break
    else: raise RuntimeError( "Could not generate NaN‑free validation data." )
    ValidationDict[ "y" ].append( y_val )
    ValidationDict[ "Data" ].append( [ x_val ] )

  # ---- 5. Run the Arborescence ---------------------------------------------
  Arbo = NARMAX.Arborescence(
        y,
        Ds = None,
        DsNames = None,
        Dc = RegMat,
        DcNames = RegNames,
        tolRoot = tol,
        tolRest = tol,
        MaxDepth = ArboDepth,
        ValFunc = NARMAX.DefaultValidation,
        ValData = ValidationDict,
        Verbose = False,
    )
  Arbo.fit()
  theta, L, ERR, Morphdict, RegMat, RegNames = Arbo.get_Results()

  # ---- Type, shape, and dtype checks --------------
  assert isinstance( theta, tor.Tensor ), f"Expected torch.Tensor, got {type( theta )}"
  assert isinstance( L, np.ndarray ), f"Expected np.ndarray, got {type( L )}"
  assert isinstance( ERR, np.ndarray ), f"Expected np.ndarray, got {type( ERR )}"
  assert isinstance( Morphdict, ( dict, type( None ) ) ), f"Expected dict or None, got {type( Morphdict )}"
  assert isinstance( RegMat, tor.Tensor ), f"Expected torch.Tensor, got {type( RegMat )}"
  assert isinstance( RegNames, np.ndarray ), f"Expected np.ndarray, got {type( RegNames )}"

  assert theta.ndim == 1, f"Expected 1D theta, got {theta.ndim}D"
  assert L.ndim == 1, f"Expected 1D L, got {L.ndim}D"
  assert ERR.ndim == 1, f"Expected 1D ERR, got {ERR.ndim}D"
  assert RegMat.ndim == 2, f"Expected 2D RegMat, got {RegMat.ndim}D"
  assert RegNames.ndim == 1, f"Expected 1D RegNames, got {RegNames.ndim}D"

  assert L.dtype == np.int64, f"Expected int64 L, got {L.dtype}"
  assert ERR.dtype == np.float64, f"Expected float64 ERR, got {ERR.dtype}"

  nr = len( theta )
  assert nr == len( L ) == len( ERR ), f"Inconsistent lengths: theta={nr}, L={len( L )}, ERR={len( ERR )}"
  assert RegMat.shape[ 1 ] == len( RegNames ), f"RegMat cols {RegMat.shape[ 1 ]} != RegNames {len( RegNames )}"

  assert np.all( L >= 0 ), "L contains negative indices"
  assert np.all( L < RegMat.shape[ 1 ] ), f"L indices out of range: max L={L.max()}, RegMat cols={RegMat.shape[ 1 ]}"
  assert tor.all( tor.isfinite( theta ) ), "theta contains non-finite values"
  assert np.all( np.isfinite( ERR ) ), "ERR contains non-finite values"
  assert np.all( ERR >= 0 ), "ERR must be non-negative"

  # ---- Sanity checks ------------------------------
  assert len( L ) != 0, "Expected no dictionary candidates"
  assert len( theta ) != 0, "Expected non-empty theta"
  assert len( ERR ) != 0, "Expected non-empty ERR"
  assert len( RegMat ) != 0, "Expected non-empty RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert len( RegNames ) != 0, "Expected non-empty RegNames"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx = np.argsort( L )
  L_sorted = np.asarray( L )[ sort_idx ]
  theta_sorted = theta[ sort_idx ].cpu().numpy()
  names_sorted = np.asarray( RegNames )[ L_sorted ] # select & order names

  # ---- 7. Expected values (known example results) ---------------------------------------
  expected_theta = np.array( [ 0.3, 0.3, -0.4, 0.7, -0.4, 0.5, -0.5 ] )
  expected_L = np.array( [ 0, 99, 215, 310, 339, 458, 675 ] )
  expected_names = np.array(
        [ 'x[k]', 'x[k-1]^3', 'y[k-3]^3', 'abs(x[k-1]^2 * x[k-2])', 'abs(x[k-1] * y[k-2]^2)', 'cos(x[k-2] * y[k-1])',
         'exp(x[k-2] * x[k-3])' ]
    )

  # ---- 8. Assertions -------------------------------------------------------
  np.testing.assert_array_equal( L_sorted, expected_L )
  np.testing.assert_array_equal( names_sorted, expected_names )
  np.testing.assert_allclose( theta_sorted, expected_theta, rtol = 1e-5, atol = 1e-7 )

  # ---- Additional sorted-value checks -----------------------------------
  assert np.array_equal( L_sorted, np.sort( L ) ), "L_sorted is incorrectly sorted"
  assert np.all( np.diff( L_sorted ) > 0 ), "L_sorted must be strictly increasing"

  # ---- Check Arborescence object attributes after fit -------------------
  assert Arbo.theta is not None, "Arbo.theta should be set after fit"
  assert Arbo.L is not None, "Arbo.L should be set after fit"
  assert Arbo.ERR is not None, "Arbo.ERR should be set after fit"


  # ------------------- Verification on fresh test data -------------------
  # TODO


def test_nonlinearity_construction() -> None:
  '''Test NonLinearity object construction and attribute access.'''
  abs_nl = NARMAX.NonLinearity( "abs", f = tor.abs )
  assert abs_nl.Name == "abs"
  assert abs_nl.f is tor.abs
  assert abs_nl.get_Name() == "abs"

  cos_nl = NARMAX.NonLinearity( "cos", f = tor.cos )
  assert cos_nl.Name == "cos"
  assert cos_nl.f is tor.cos
  assert cos_nl.get_Name() == "cos"

  exp_nl = NARMAX.NonLinearity( "exp", f = tor.exp )
  assert exp_nl.Name == "exp"
  assert exp_nl.f is tor.exp
  assert exp_nl.get_Name() == "exp"

  assert NARMAX.Identity.Name == "id"
  assert callable( NARMAX.Identity.f )


def test_validation_dict_structure() -> None:
  '''Test that a correctly formed ValidationDict has the expected keys and types.'''
  vd: dict = {
    "y": [],
    "Data": [],
    "InputVarNames": [ "x", "y" ],
    "NonLinearities": [ NARMAX.Identity ],
  }
  assert "y" in vd
  assert "Data" in vd
  assert "InputVarNames" in vd
  assert "NonLinearities" in vd
  assert isinstance( vd[ "y" ], list )
  assert isinstance( vd[ "Data" ], list )
  assert isinstance( vd[ "InputVarNames" ], list )
  assert all( isinstance( n, str ) for n in vd[ "InputVarNames" ] )
  assert isinstance( vd[ "NonLinearities" ], list )


def test_arborescence_default_parameters() -> None:
  '''Test Arborescence default parameter values before fitting.'''
  arbo = NARMAX.Arborescence()
  assert arbo.tolRoot == 0.001
  assert arbo.tolRest == 0.001
  assert arbo.MaxDepth == 5
  assert arbo.Verbose is False
  assert arbo.theta is None
  assert arbo.L is None
  assert arbo.ERR is None
  assert arbo.MorphDict is None
  assert arbo.y is None
