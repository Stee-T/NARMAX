import torch as tor
import numpy as np
import matplotlib
matplotlib.use("Agg")

import NARMAX
import NARMAX.TestSystems as Test_Systems


def test_rational_nonlinear_system() -> None:
  '''Integration test for rational nonlinear system identification.'''
  # Reproducibility
  tor.manual_seed(42)  # for tor.rand in validation/test
  np.random.seed(252)  # for the RNG used to create training data

  # ---------------------------------------------------- 2. Hyper-parameters
  p = 2_500
  InputAmplitude = 1.0
  tol = 0.0001
  W = None
  qx = 3
  qy = 3
  ExpansionOrder = 3
  ArboDepth = 3

  Sys = Test_Systems.RatNonLinSystem
  assert callable(Sys)

  # Deterministic training data using the original RNG method
  RNG = np.random.default_rng(seed=2520)
  x = tor.tensor((2 * InputAmplitude) * (RNG.random(p) - 0.5))
  x -= tor.mean(x)
  assert x.ndim == 1, "Expected x to be 1D"
  x, y, W = Sys(x, W, Print=False)
  assert W is None, "Expected W to be None (no noise passed)"
  assert len(y) == p - 2, f"Expected y to have {p - 2} samples after Sys (MaxLag=2), got {len(y)}"
  if (tor.isnan(tor.sum(y))):
    raise AssertionError("Training data contains NaN")

  NonLinearities = [NARMAX.Identity]
  MakeRational = [True]
  NonLinearities.append(NARMAX.NonLinearity("abs", f=tor.abs))
  MakeRational.append(True)

  # Check NonLinearities structure
  assert len(NonLinearities) == 2
  for i, nl in enumerate(NonLinearities):
    assert isinstance(nl, NARMAX.NonLinearity), f"NonLinearities[{i}] must be NonLinearity"
  assert NonLinearities[0].get_Name() == "id"
  assert NonLinearities[1].get_Name() == "abs"
  assert NonLinearities[0] is NARMAX.Identity, "First should be the shared Identity object"

  # Check MakeRational structure
  assert len(MakeRational) == 2
  assert all(isinstance(mr, bool) for mr in MakeRational)
  assert all(MakeRational)

  # ---------------------------------------------------- 3. Training Data
  y, RegMat, RegNames = NARMAX.CTors.Lagger(Data=(x, y), Lags=(qx, qy))
  RegMat, RegNames = NARMAX.CTors.Expander(RegMat, RegNames, ExpansionOrder)
  RegMat, RegNames, _ = NARMAX.CTors.NonLinearizer(
        y, RegMat, RegNames, NonLinearities, MakeRational=MakeRational
    )

  assert isinstance(RegMat, tor.Tensor), "Expected RegMat to be torch.Tensor"
  assert RegMat.ndim == 2, f"Expected RegMat to be 2D, got {RegMat.ndim}D"
  assert isinstance(RegNames, np.ndarray), "Expected RegNames to be numpy array"
  assert RegNames.ndim == 1, f"Expected RegNames to be 1D, got {RegNames.ndim}D"
  assert np.issubdtype(RegNames.dtype, np.str_), f"Expected RegNames to be str array, got {RegNames.dtype}"
  assert RegMat.shape[1] == len(RegNames), \
      f"RegMat columns ({RegMat.shape[1]}) != RegNames length ({len(RegNames)})"

  # ---------------------------------------------------- 4. Validation Data
  ValidationDict = {
        "y": [],
        "Data": [],
        "InputVarNames": ["x", "y"],
        "DcNames": RegNames,
        "NonLinearities": NonLinearities,
    }

  # Check ValidationDict structure
  expected_keys = {"y", "Data", "InputVarNames", "DcNames", "NonLinearities"}
  assert set(ValidationDict.keys()) == expected_keys
  assert ValidationDict["InputVarNames"] == ["x", "y"]
  assert ValidationDict["NonLinearities"] is NonLinearities
  assert ValidationDict["DcNames"] is RegNames

  for _ in range(5):
    max_attempts = 100
    for _ in range(max_attempts):
      x_val = tor.rand(int(p / 3))
      x_val -= tor.mean(x_val)
      x_val, y_val, W = Sys(x_val, W, Print=False)
      if (not tor.isnan(tor.sum(y_val))):
        break
    ValidationDict["y"].append(y_val)
    ValidationDict["Data"].append([x_val])

  # Check validation data structure
  assert len(ValidationDict["y"]) == 5
  assert len(ValidationDict["Data"]) == 5
  for i in range(5):
    assert isinstance(ValidationDict["y"][i], tor.Tensor), \
        f"Validation y[{i}] must be torch.Tensor"
    assert isinstance(ValidationDict["Data"][i], list) and len(ValidationDict["Data"][i]) == 1, \
        f"Validation Data[{i}] structure unexpected"
    assert isinstance(ValidationDict["Data"][i][0], tor.Tensor), \
        f"Validation Data[{i}][0] must be torch.Tensor"

  # ---------------------------------------------------- 5. Running the Arborescence
  Arbo = NARMAX.Arborescence(
        y,
        Ds=None,
        DsNames=None,
        Dc=RegMat,
        DcNames=RegNames,
        tolRoot=tol,
        tolRest=tol,
        MaxDepth=ArboDepth,
        ValFunc=NARMAX.DefaultValidation,
        ValData=ValidationDict,
        Verbose=False,
    )
  assert isinstance(Arbo, NARMAX.Arborescence)

  Arbo.fit()
  theta, L, ERR, Morphdict, RegMat, RegNames = Arbo.get_Results()

  # ---- Sanity checks ------------------------------
  assert len(L) != 0, "Expected no dictionary candidates"
  assert len(theta) != 0, "Expected non-empty theta"
  assert len(ERR) != 0, "Expected non-empty ERR"
  assert len(RegMat) != 0, "Expected non-empty RegMat"
  assert Morphdict is None, "Expected empty Morphdict"
  assert len(RegNames) != 0, "Expected non-empty RegNames"

  # ---- Additional type/shape checks ---------------
  assert isinstance(theta, tor.Tensor), f"Expected theta to be torch.Tensor, got {type(theta)}"
  assert tor.is_floating_point(theta), "Expected theta to be a floating-point tensor"
  assert theta.ndim == 1, f"Expected theta to be 1D, got {theta.ndim}D"

  assert isinstance(L, np.ndarray), f"Expected L to be numpy array, got {type(L)}"
  assert L.dtype == np.int64, f"Expected L to be int64, got {L.dtype}"
  assert L.ndim == 1, f"Expected L to be 1D, got {L.ndim}D"

  assert isinstance(ERR, np.ndarray), f"Expected ERR to be numpy array, got {type(ERR)}"
  assert np.issubdtype(ERR.dtype, np.floating), f"Expected ERR to be float, got {ERR.dtype}"
  assert ERR.ndim == 1, f"Expected ERR to be 1D, got {ERR.ndim}D"
  assert np.all(ERR >= 0), "Expected all ERR values to be non-negative"
  assert ERR[0] > 0, "First ERR value should be positive (best term)"

  assert isinstance(RegMat, tor.Tensor), "Expected returned RegMat to be torch.Tensor"
  assert RegMat.ndim == 2, f"Expected returned RegMat to be 2D, got {RegMat.ndim}D"

  assert isinstance(RegNames, np.ndarray), "Expected returned RegNames to be numpy array"
  assert RegNames.ndim == 1, "Expected returned RegNames to be 1D"
  assert np.issubdtype(RegNames.dtype, np.str_), \
      f"Expected returned RegNames to be str array, got {RegNames.dtype}"

  # ---- Consistency checks ------------------------
  assert len(L) == len(theta) == len(ERR), \
      f"Length mismatch: L={len(L)}, theta={len(theta)}, ERR={len(ERR)}"
  assert RegMat.shape[1] == len(RegNames), \
      f"Returned RegMat columns ({RegMat.shape[1]}) != RegNames ({len(RegNames)})"

  # ---- 6. Sort by L to obtain order independent of ERR ------------------------------
  sort_idx = np.argsort(L)
  L_sorted = np.asarray(L)[sort_idx]
  theta_sorted = theta[sort_idx].cpu().numpy()
  names_sorted = np.asarray(RegNames)[L_sorted]  # select & order names

  # ---- 7. Expected values (known example results) ---------------------------------------
  expected_theta = np.array([-0.3, -0.35, 0.6, 0.1, 0.3, -0.2, -0.4, 0.3])
  expected_L = np.array([18, 35, 119, 123, 253, 294, 350, 357])
  expected_names = np.array(['x[k-1] * y[k-2]', 'x[k]^3', 'abs(x[k])', 'abs(y[k-1])',
       '~/(x[k-2] * y[k-1])', '~/(x[k-1]^3)', '~/abs(x[k])', '~/abs(x[k] * x[k-1])']
    )

  # ---- 8. Assertions -------------------------------------------------------
  np.testing.assert_array_equal(L_sorted, expected_L)
  np.testing.assert_array_equal(names_sorted, expected_names)
  np.testing.assert_allclose(theta_sorted, expected_theta, rtol=1e-5, atol=1e-7)

  # ---- 9. Verification on fresh test data via SymbolicOscillator -----------
  # Generate fresh test data with a new seed
  tor.manual_seed(99999)
  TestInput = (2 * InputAmplitude) * (tor.rand(p) - 0.5)
  TestInput -= TestInput.mean()
  _, y_true, _ = Sys(TestInput, None, Print=False)

  # Build the model from the selected terms
  Model = NARMAX.SymbolicOscillator(
      ["x", "y"], NonLinearities, RegNames[L], theta
    )
  assert isinstance(Model, NARMAX.SymbolicOscillator), "Expected model to be SymbolicOscillator"

  # Prefill the internal state to match the ground truth
  Model.set_InputStorage(TestInput[None, :Model.get_MaxInputLag()].clone())
  Model.set_OutputStorage(y_true[:Model.get_MaxOutputLag()].clone())
  yHat = Model.Oscillate([TestInput[-len(y_true):]])

  assert isinstance(yHat, tor.Tensor), "Expected yHat to be a torch.Tensor"
  assert len(yHat) == len(y_true), \
      f"Length mismatch: yHat={len(yHat)}, y_true={len(y_true)}"

  # Compare ignoring the first few transient samples
  skip = 10
  error = (yHat[skip:].cpu() - y_true[skip:].cpu()).abs()
  max_error = error.max().item()
  rmse = tor.sqrt((error ** 2).mean()).item()
  assert max_error < 0.01, f"Model output max error {max_error:.6f} exceeds 0.01"
  assert rmse < 0.001, f"Model output RMSE {rmse:.6f} exceeds 0.001"
