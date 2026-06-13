import copy
import torch as tor
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt

import NARMAX
import NARMAX.TestSystems as Test_Systems

def test_three_input_mimo() -> None:
  '''Integration test for a three-input MIMO system.'''
  # Reproducibility
  tor.manual_seed( 42045 )
  np.random.seed( 42045 )

  # ------------------- Inputs creation -------------------
  p = 2_000
  ExpansionOrder = 3
  InputAmplitude = 1.5
  tol = 1e-9
  ArboDepth = 4
  W = None
  Lags = ( 3, 3, 3, [ 1, 2, 3 ], [ 1, 2, 3 ] ) # x1, x2, x3, y1, y2, exclude y1[k], y2[k]

  max_attempts = 100
  while ( True ):
    x1 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
    x2 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
    x3 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
    x1 -= tor.mean( x1 )
    x2 -= tor.mean( x2 )
    x3 -= tor.mean( x3 )
    x1, x2, x3, y1, y2, W = Test_Systems.ThreeInputMIMO( x1, x2, x3, W, Print = False )
    if ( not ( tor.isnan( tor.sum( y1 ) ) or tor.isnan( tor.sum( y2 ) ) ) ): break

  NonLinearities = [ NARMAX.Identity, NARMAX.NonLinearity( "abs", f = tor.abs ) ]
  InputVars = [ "x1", "x2", "x3", "y1", "y2" ]

  # Training data
  _, RegMat, RegNames = NARMAX.CTors.Lagger( Data = [ x1, x2, x3, y1, y2 ], Lags = Lags,
                                               RegNames = InputVars
                                             )
  RegMat, RegNames = NARMAX.CTors.Expander( RegMat, RegNames, ExpansionOrder )
  RegMat, RegNames, _ = NARMAX.CTors.NonLinearizer( None, RegMat, RegNames, NonLinearities )

  y1_cut = NARMAX.CutY( y1, Lags )
  y2_cut = NARMAX.CutY( y2, Lags )

  # Validation data
  ValidationDict1 = {
        "y": [],
        "Data": [],
        "InputVarNames": [ "x1", "x2", "x3", "y2", "y1" ],
        "NonLinearities": NonLinearities,
        "OutputVarName": "y1"
    }

  ValidationDict2 = copy.deepcopy( ValidationDict1 )
  ValidationDict2[ "OutputVarName" ] = "y2"
  ValidationDict2[ "InputVarNames" ] = InputVars

  for i in range( 5 ):
    for _ in range( max_attempts ):
      x1_val = tor.rand( int( p / 3 ) )
      x2_val = tor.rand( int( p / 3 ) )
      x3_val = tor.rand( int( p / 3 ) )
      x1_val -= tor.mean( x1_val )
      x2_val -= tor.mean( x2_val )
      x3_val -= tor.mean( x3_val )
      x1_val, x2_val, x3_val, y1_val, y2_val, W = Test_Systems.ThreeInputMIMO(
                x1_val, x2_val, x3_val, W, Print = False
            )
      if ( not ( tor.isnan( tor.sum( y1_val ) ) or tor.isnan( tor.sum( y2_val ) ) ) ): break
    ValidationDict1[ "y" ].append( y1_val )
    ValidationDict2[ "y" ].append( y2_val )
    ValidationDict1[ "Data" ].append( [ x1_val, x2_val, x3_val, y2_val ] )
    ValidationDict2[ "Data" ].append( [ x1_val, x2_val, x3_val, y1_val ] )

  # Fit model for y1
  Arbo_1 = NARMAX.Arborescence(
        y1_cut,
        Dc = RegMat, DcNames = RegNames,
        tolRoot = tol, tolRest = tol,
        MaxDepth = ArboDepth,
        ValFunc = NARMAX.DefaultValidation,
        ValData = ValidationDict1,
    )
  theta1, L1, ERR1, Morphdict1, RegMat1, RegNames1 = Arbo_1.fit()

  # Fit model for y2
  Arbo_2 = NARMAX.Arborescence(
        y2_cut,
        Dc = RegMat, DcNames = RegNames,
        tolRoot = tol, tolRest = tol,
        MaxDepth = ArboDepth,
        ValFunc = NARMAX.DefaultValidation,
        ValData = ValidationDict2,
    )
  theta2, L2, ERR2, Morphdict2, RegMat2, RegNames2 = Arbo_2.fit()

  # ---- 5. Sanity checks ------------------------------
  assert len( L1 ) != 0, "Expected non-empty L"
  assert len( L2 ) != 0, "Expected non-empty L"
  assert len( theta1 ) != 0, "Expected non-empty theta"
  assert len( theta2 ) != 0, "Expected non-empty theta"
  assert len( ERR1 ) != 0, "Expected non-empty ERR"
  assert len( ERR2 ) != 0, "Expected non-empty ERR"
  assert len( RegMat1 ) != 0, "Expected non-empty RegMat"
  assert len( RegMat2 ) != 0, "Expected non-empty RegMat"
  assert Morphdict1 is None, "Expected empty Morphdict"
  assert Morphdict2 is None, "Expected empty Morphdict"
  assert len( RegNames1 ) != 0, "Expected non-empty RegNames"
  assert len( RegNames2 ) != 0, "Expected non-empty RegNames"

  # ---- 5b. Type checks ------------------------------
  assert isinstance( theta1, tor.Tensor ), "theta1 must be a torch.Tensor"
  assert isinstance( theta2, tor.Tensor ), "theta2 must be a torch.Tensor"
  assert isinstance( L1, np.ndarray ), "L1 must be a numpy array"
  assert isinstance( L2, np.ndarray ), "L2 must be a numpy array"
  assert isinstance( ERR1, np.ndarray ), "ERR1 must be a numpy array"
  assert isinstance( ERR2, np.ndarray ), "ERR2 must be a numpy array"
  assert isinstance( RegMat1, tor.Tensor ), "RegMat1 must be a torch.Tensor"
  assert isinstance( RegMat2, tor.Tensor ), "RegMat2 must be a torch.Tensor"
  assert isinstance( RegNames1, np.ndarray ), "RegNames1 must be a numpy array"
  assert isinstance( RegNames2, np.ndarray ), "RegNames2 must be a numpy array"

  # ---- 5c. Dtype checks ------------------------------
  assert theta1.dtype in ( tor.float32, tor.float64 ), "theta1 must be float"
  assert theta2.dtype in ( tor.float32, tor.float64 ), "theta2 must be float"
  assert L1.dtype == np.int64, "L1 must be int64"
  assert L2.dtype == np.int64, "L2 must be int64"
  assert ERR1.dtype == np.float64, "ERR1 must be float64"
  assert ERR2.dtype == np.float64, "ERR2 must be float64"
  assert RegMat1.dtype in ( tor.float32, tor.float64 ), "RegMat1 must be float"
  assert RegMat2.dtype in ( tor.float32, tor.float64 ), "RegMat2 must be float"

  # ---- 5d. Shape and dimension consistency -----------
  n_terms1 = len( theta1 )
  n_terms2 = len( theta2 )
  assert len( L1 ) == n_terms1, "L1 and theta1 must have same length"
  assert len( ERR1 ) == n_terms1, "ERR1 and theta1 must have same length"
  assert len( L2 ) == n_terms2, "L2 and theta2 must have same length"
  assert len( ERR2 ) == n_terms2, "ERR2 and theta2 must have same length"
  assert RegMat1.shape[ 0 ] == len( y1_cut ), f"RegMat1 rows ({ RegMat1.shape[ 0 ] }) must match y1_cut ({ len( y1_cut ) })"
  assert RegMat2.shape[ 0 ] == len( y2_cut ), f"RegMat2 rows ({ RegMat2.shape[ 0 ] }) must match y2_cut ({ len( y2_cut ) })"
  assert RegMat1.shape[ 1 ] == len( RegNames1 ), f"RegMat1 columns ({ RegMat1.shape[ 1 ] }) must match RegNames1 ({ len( RegNames1 ) })"
  assert RegMat2.shape[ 1 ] == len( RegNames2 ), f"RegMat2 columns ({ RegMat2.shape[ 1 ] }) must match RegNames2 ({ len( RegNames2 ) })"

  # ---- 5e. ERR value range ---------------------------
  assert np.all( ERR1 >= 0 ) and np.all( ERR1 <= 1 ), "ERR1 values must be in [0, 1]"
  assert np.all( ERR2 >= 0 ) and np.all( ERR2 <= 1 ), "ERR2 values must be in [0, 1]"

  # ---- 5f. Theta finiteness --------------------------
  assert not tor.any( tor.isnan( theta1 ) ), "theta1 must not contain NaN"
  assert not tor.any( tor.isinf( theta1 ) ), "theta1 must not contain Inf"
  assert not tor.any( tor.isnan( theta2 ) ), "theta2 must not contain NaN"
  assert not tor.any( tor.isinf( theta2 ) ), "theta2 must not contain Inf"

  # ---- 5g. L index bounds ---------------------------
  assert np.all( L1 >= 0 ), "All L1 indices must be non-negative"
  assert np.all( L2 >= 0 ), "All L2 indices must be non-negative"
  assert np.all( L1 < RegMat1.shape[ 1 ] ), "All L1 indices must be within RegMat1 columns"
  assert np.all( L2 < RegMat2.shape[ 1 ] ), "All L2 indices must be within RegMat2 columns"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx1 = np.argsort( L1 )
  L_sorted1: NDArray[ np.int64 ] = np.asarray( L1 )[ sort_idx1 ]
  theta_sorted1: NDArray[ np.float64 ] = theta1[ sort_idx1 ].cpu().numpy()
  names_sorted1: NDArray[ np.str_ ] = np.asarray( RegNames1 )[ L_sorted1 ]

  sort_idx2 = np.argsort( L2 )
  L_sorted2: NDArray[ np.int64 ] = np.asarray( L2 )[ sort_idx2 ]
  theta_sorted2: NDArray[ np.float64 ] = theta2[ sort_idx2 ].cpu().numpy()
  names_sorted2: NDArray[ np.str_ ] = np.asarray( RegNames2 )[ L_sorted2 ]

  # ---- 6b. Sorted L monotonicity --------------------
  assert np.all( np.diff( L_sorted1 ) >= 0 ), "L_sorted1 must be monotonically increasing"
  assert np.all( np.diff( L_sorted2 ) >= 0 ), "L_sorted2 must be monotonically increasing"

  # ---- 7. Expected values (known example results) ---------------------------------------
  expected_theta1: NDArray[ np.float64 ] = np.array( [ 0.2, 0.5, -0.7, 0.3, -0.3, 0.7, -0.8 ] )
  expected_L1: NDArray[ np.int64 ] = np.array( [ 0, 58, 422, 769, 974, 1337, 1467 ] )
  expected_names1: NDArray[ np.str_ ] = np.array( [ 'x1[k]', 'x1[k-2] * x2[k-3]', 'x1[k-1] * x2[k-1]^2', 'x2[k]^3',
                                                  'x2[k-2]^2 * y2[k-1]', 'abs(x3[k])', 'abs(x3[k-1] * y1[k-2])' ], dtype = '<U32' )

  expected_theta2: NDArray[ np.float64 ] = np.array( [ 0.3, 0.6, -0.7, 0.5, -0.4, 0.7, -0.9 ] )
  expected_L2: NDArray[ np.int64 ] = np.array( [ 1, 65, 878, 1109, 1211, 1341, 1470 ] )
  expected_names2: NDArray[ np.str_ ] = np.array( [ 'x1[k-1]', 'x1[k-2] * y1[k-3]', 'x2[k-1]^2 * x3[k-1]', 'x3[k]^3',
                                                  'x3[k-2]^2 * y1[k-1]', 'abs(y1[k-1])', 'abs(x3[k-1] * y2[k-2])' ], dtype = '<U32' )

  # ---- 8. Assertions -------------------------------------------------------
  np.testing.assert_array_equal( L_sorted1, expected_L1 )
  np.testing.assert_array_equal( names_sorted1, expected_names1 )
  np.testing.assert_allclose( theta_sorted1, expected_theta1, rtol = 1e-4, atol = 1e-4 )

  np.testing.assert_array_equal( L_sorted2, expected_L2 )
  np.testing.assert_array_equal( names_sorted2, expected_names2 )
  np.testing.assert_allclose( theta_sorted2, expected_theta2, rtol = 1e-4, atol = 1e-4 )

  # ---- 9. Verification on fresh test data -------------------
  # Generate one fresh test fold independent of the validation data
  W = None  # reset system state for a fresh test
  for _ in range( max_attempts ):
    x1_test = tor.rand( int( p / 3 ) )
    x2_test = tor.rand( int( p / 3 ) )
    x3_test = tor.rand( int( p / 3 ) )
    x1_test -= tor.mean( x1_test )
    x2_test -= tor.mean( x2_test )
    x3_test -= tor.mean( x3_test )
    x1_test, x2_test, x3_test, y1_test, y2_test, W = Test_Systems.ThreeInputMIMO(
        x1_test, x2_test, x3_test, W, Print = False
    )
    if ( not ( tor.isnan( tor.sum( y1_test ) ) or tor.isnan( tor.sum( y2_test ) ) ) ): break

  # Build test validation dicts matching the structure used during fitting
  TestDict1 = {
      "y": [ y1_test ],
      "Data": [ [ x1_test, x2_test, x3_test, y2_test ] ],
      "InputVarNames": [ "x1", "x2", "x3", "y2", "y1" ],
      "NonLinearities": NonLinearities,
      "OutputVarName": "y1"
  }
  TestDict2 = {
      "y": [ y2_test ],
      "Data": [ [ x1_test, x2_test, x3_test, y1_test ] ],
      "InputVarNames": InputVars,
      "NonLinearities": NonLinearities,
      "OutputVarName": "y2"
  }

  # Selected regressor names (Ds is empty, so just the candidate names at indices L)
  chosen_names1: NDArray[ np.str_ ] = RegNames1[ L1 ]
  chosen_names2: NDArray[ np.str_ ] = RegNames2[ L2 ]

  # Compute the same relative MAE metric used during validation
  error1: float = NARMAX.DefaultValidation( theta1, L1, ERR1, chosen_names1, TestDict1 )
  error2: float = NARMAX.DefaultValidation( theta2, L2, ERR2, chosen_names2, TestDict2 )

  # Each error is the mean relative absolute error; a correctly identified model should be well below 50%
  assert error1 < 0.5, f"y1 relative MAE on fresh test data too high: { error1 }"
  assert error2 < 0.5, f"y2 relative MAE on fresh test data too high: { error2 }"
