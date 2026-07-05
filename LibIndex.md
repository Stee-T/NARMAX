# NARMAX — Library Index

## 1. Library Overview

- **Purpose**: Machine-learning library for NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous inputs) symbolic system identification. Implements the AOrLSR (Arborescent Orthogonal Least Squares with imposed Regressors) algorithm — a breadth-first tree search over regressor subsets using recursive Forward Orthogonal Least Squares (rFOrLSR) with ERR-based pruning, OOIT (One-step-ahead Output In The loop) prediction caching, and dictionary morphing (nonlinear function optimization via scipy trust-region methods).
- **Architecture**: Four-layer design — (1) **CTors** module builds the regressor dictionary (lagging, monomial expansion, nonlinear transforms, rational extensions, boolean operations); (2) **Classes** provides core data structures (`Arborescence` orchestrates the BFS tree traversal, `SymbolicOscillator` evaluates NARMAX models, `MultiKeyHashTable`/`Queue` support the search); (3) **Tools** offers analysis utilities (IIR conversion, pole-zero plots, lag/order estimation); (4) **TestSystems** provides ground-truth NARMAX benchmarks. The `Arborescence.fit()` method is the main entry point.
- **Bootstrap**: `NARMAX.HelperFuncs.Set_Tensortype_And_Device()` detects CUDA/MPS/CPU and sets `torch` defaults. `NARMAX.Identity` is a pre-built identity `NonLinearity`. The `SymbolicOscillator` module-level `Device` string is computed once at import.
- **Dependencies**: `numpy`, `scipy`, `matplotlib`, `dill` (serialization), `tqdm` (progress bars), `torch` (≥1.12). Python ≥3.12.

## 2. Data Types & Interfaces

### `NARMAX.Classes.NonLinearity` — `NonLinearity`
- **Purpose**: Wraps a scalar nonlinear function with optional first/second derivatives for dictionary morphing.
- **Fields**:
  - `Name: str` — identifier
  - `f: Callable[[tor.Tensor], tor.Tensor]` — the function
  - `fPrime: Optional[Callable]` — first derivative
  - `fSecond: Optional[Callable]` — second derivative
  - `ToMorph: bool` — whether this function is eligible for dictionary morphing
- **Constructor** validates: callable, same-shape output, no NaNs on test input, elementwise (no time-dependence). If `fPrime`/`fSecond` absent, `ToMorph` is forced `False`.

### `NARMAX.Classes.Parser_0_3` — Data Classes

#### `SubExpression` (dataclass)
- `Coeff: Optional[float]` — multiplicative coefficient
- `VarName: Optional[str]` — variable name
- `Lag: Optional[int]` — discrete-time lag
- `Exponent: Optional[float]` — power exponent
- `__str__()` renders as e.g. `0.5*x[k-1]^2`

#### `ParsedReg` (dataclass)
- `FuncName: Optional[str]` — nonlinear function name (e.g. `"abs"`, `"~/"` for rational)
- `SubExpressions: list[SubExpression]` — argument list
- `Operators: list[str]` — operators between sub-expressions (`+`, `-`, `*`, `/`)
- `__str__()` renders full structured representation

### `NARMAX.Classes.Queue` — `Queue`
- **Purpose**: Bounded FIFO wrapping `collections.deque`. Accepts only integer `ndarray` items.
- **Methods**: `put()`, `get()` (raises `QueueEmptyError` if empty), `is_empty()`, `clear()`, `size()`, `peek()` (returns first element or empty array).

### `NARMAX.Classes.MultiKeyHashTable` — `MultiKeyHashTable` (original)
- **Purpose**: Look-up table mapping unordered integer subsets to Data-list indices. Used by `Arborescence` for OOIT caching.
- `Data: list[NDArray[np.int64]]` — stored sequences
- `LookUpDict: dict[tuple[int], int]` — sorted-tuple → index
- **Methods**: `SameStart(Item)`, `AddData(Item)`, `CreateKeys(MinLen, IndexSet, Value)`, `DeleteAllOfSize(n)`

### `NARMAX.Classes.New MultiKeyHashMap` — `MultiKeyHashTable` (performance-optimized)
- **Purpose**: Drop-in replacement using length-partitioned open-addressing hash tables with contiguous `NDArray` storage. O(1) level deletion.
- `int_type: np.dtype`, `tables: list[_LevelTable | None]`, `_flat_data: NDArray`, `_data_count: int`
- **Methods**: same API — `SameStart()`, `AddData()`, `CreateKeys()`, `DeleteAllOfSize()`. Uses commutative hash (`0x9e3779b97f4a7c15` magic constant) with 75% load-factor resize.

### `NARMAX.Classes.Arborescence` — `Arborescence`
- **Purpose**: Main orchestrator — breadth-first search over NARMAX regressor subsets.
- **Constructor parameters**: `y`, `Ds`, `DsNames`, `Dc`, `DcNames`, `tolRoot`, `tolRest`, `MaxDepth`, `ValFunc`, `ValData`, `Verbose`, `MorphDict`, `U`, `FileName`, `SaveFrequency`.
- **Key fields**: `Q` (Queue), `LG` (MultiKeyHashTable), `MinLen`, `TotalNodes`, `nNotSkippedNodes`, `theta`, `ERR`, `L`.
- **Methods**: `fit()`, `validate()`, `get_Results()`, `rFOrLSR()`, `set_ArboDepth()`, `PlotAndPrint()`, `load()`, `MemoryDump()`.

### `NARMAX.Classes.SymbolicOscillator` — `SymbolicOscillator`
- **Purpose**: Compiled NARMAX model evaluator — parses symbolic expression strings into `eval`-generated system lambdas (MA numerator/denominator, AR numerator/denominator, combined system).
- **Constructor**: Takes `InputVarNames`, `NonLinearities`, `ExprList` (strings), `theta`, `OutputVarName`, `dtype`, `device`. Validates names, deduplicates, parses expressions, finds max lags, creates storage buffers.
- **Key fields**: `theta`, `SubSystem_MA_Num/Den`, `System_BufferStart/Main`, `InputStorage`, `OutputStorage`, `MaxInputLag`, `MaxOutputLag`, `MaxPositiveInputLag`.
- **Methods**: `Oscillate(Data, theta?, DsData?)`, `set_theta/get_theta`, `set_InputStorage/get_InputStorage`, `set_OutputStorage/get_OutputStorage`, `zeroInternalStorage`, `get_nRegressors`, `get_nInputVars`, `get_MaxInputLag/get_MaxOutputLag/get_MaxPositiveInputLag`.

### `NARMAX.Classes.Morpher` — Module-level functions
- **Purpose**: Dictionary morphing — continuous optimization of nonlinear regressor arguments using trust-region methods.
- **Key functions**: `DictionaryMorpher()`, `GenVSGen()` (brute-force vector-space generator), `InfOPT()` (scipy `trust-exact` refinement), `DMFunc/DMGrad/DMHessian` (cost, gradient, Hessian for correlation-based morphing objective), `IsMorphable()` (tests relaxed homogeneity), `Expressionparser()` (index → function).

## 3. Module & File Index

### `NARMAX/` (top-level package)

#### `NARMAX/__init__.py`
- **Responsibility**: Package initialization — exposes public API and pre-built constants.
- **Constants**: `Identity = NonLinearity("id", lambda x: x)` — identity function object.
- **Public API**:
  - Sub-namespace imports: `NARMAX.Tools`, `NARMAX.CTors`
  - Direct namespace: `NonLinearity`, `SymbolicOscillator`, `Device`, `Arborescence`, `CutY`, `InitAndComputeBuffer`, `DefaultValidation`

#### `NARMAX/HelperFuncs.py`
- **Responsibility**: Low-level numerical helpers (tensor ops, combinatorics, duplicate removal).
- **Public API**:
  - `Set_Tensortype_And_Device() -> str` — detects CUDA/MPS/CPU, sets `torch` default dtype (float64/float32) and device, returns device string.
  - `Combinations(N: int, k: int) -> int` — binomial coefficient via `math.comb`. Raises `TypeError`/`ValueError` for invalid input.
  - `CutY(y: tor.Tensor, Lags: int | Sequence) -> tor.Tensor` — trims front of 1-D signal `y` by max lag. Supports nested lags. Raises `TypeError`/`ValueError`.
  - `Norm2(x: tor.Tensor, epsilon: float = 1e-12) -> float | tor.Tensor` — dimension-aware squared Euclidean norm. Returns scalar for (p,) or (p,1); row-vector for (p,n). Clips below epsilon. Raises `ValueError` for scalars or NaN input.
  - `DeleteColumn(Tensor: tor.Tensor, index: int) -> tor.Tensor` — fast column deletion from 2-D tensor. Uses view for last column, `cat` otherwise. Raises `ValueError`/`IndexError`.
  - `SolveSystem(A_List: list[tor.Tensor], W: list[float]) -> tor.Tensor` — solves dense upper-unitriangular system via `torch.linalg.solve_triangular`. Strict validation on tensor shapes/devices/dtypes.
  - `RemoveDuplicates(RegMat, RegNames) -> tuple[Tensor, NDArray, NDArray]` — OOM-safe exact duplicate column removal using dual fingerprint projection + exact verification. PyTorch ≥1.12.
  - `AllCombinations(ImposedRegs, InputSeq, INT_TYPE) -> NDArray` — constructs rows `[Imposed | new_regressor]` for all valid candidates. Validates `dtype` safety and no-duplicates invariant.

#### `NARMAX/Validation.py`
- **Responsibility**: Validation procedures for selected regressor sets.
- **Public API**:
  - `InitAndComputeBuffer(Model: SymbolicOscillator, y, Data) -> tor.Tensor` — initializes model state (output/input storage) and runs `Model.Oscillate()`. Returns `yHat`.
  - `DefaultValidation(theta, L, ERR, RegNames, ValData, DcFilterIdx?) -> np.float64` — time-domain MAE validation. Builds a `SymbolicOscillator`, runs `InitAndComputeBuffer` for each validation dataset, returns mean relative MAE. Validates `ValData` dict structure (keys: `y`, `Data`, `InputVarNames`, `NonLinearities`, optional `OutputVarName`). Raises `AssertionError` on structural issues.

### `NARMAX/Classes/` (core data structures & algorithms)

#### `NARMAX/Classes/Arborescence.py`
- **Responsibility**: BFS tree traversal over regressor subsets using rFOrLSR + OOIT + ADTT (Adaptive Depth Tree Traversal). Main entry: `fit()`.
- **Public API**:
  - `__init__(...)` — see §2 for parameter list. Validates all types. Centers `y`, `Ds`, `Dc`. Runs `RemoveDuplicates` on Dc and Ds. Initializes Queue, MultiKeyHashTable, statistics. Sets default ValFunc to `1 - sum(ERR)`.
  - `MemoryDump(nComputed)` — serializes full Arborescence state via `dill` to `self.FileName`. Includes all user inputs, processing data, statistics.
  - `load(FileName, Print=True)` — deserializes and restores state. Rolls back on failure. Prints stats.
  - `rFOrLSR(y, Ds?, Dc?, U?, tol, MaxTerms=inf, OutputAll=False, LI?) -> tuple` — Recursive Forward Orthogonal Least Squares. Returns `(L,)` or `(theta, L, ERR)`. Handles Ds (imposed), Dc (candidates), dictionary morphing, OOIT prediction via LG lookup. Exits early if `Abort` flag set or if `MaxTerms` exceeded.
    - **Parameters**: `LI` is the current path for OOIT; `OutputAll=True` forces full return.
    - **Returns**: `L` (ndarray of indices), optionally `theta` (coefficients) and `ERR` (error reduction ratios).
  - `fit(FileName?, SaveFrequency?) -> tuple` — breadth-first AOrLSR traversal. Performs root regression, then iterates Queue levels. Uses ADTT: `MaxDepth = min(MaxDepth, MinLen[-1])`. Calls `validate()` at end. Supports periodic `dill` checkpointing.
    - **Returns**: `(theta, L, ERR, MorphDict, Dc, DcNames)`.
  - `validate() -> tuple` — evaluates all minimal-length regressor sequences via `rFOrLSR` + `ValFunc`. Selects best (lowest error). Returns same tuple as `get_Results()`.
  - `get_Results() -> tuple` — returns `(theta, L, ERR, MorphDict, Dc, DcNames)`. Raises `AssertionError` if no results available.
  - `set_ArboDepth(Depth: int)` — safely adjusts `MaxDepth`. Validates constraints.
  - `PlotAndPrint(ValData, PrintRegressor=True) -> tuple[Figure, Axes]` — generates 2-panel signal comparison plot and 2-panel ERR/MAE progression plot. Prints regressor terms and coefficients.
- **Errors**: `AssertionError` for invalid state (None y, uninitialized), `ValueError` for bad parameter types.

#### `NARMAX/Classes/SymbolicOscillator_0_4.py`
- **Responsibility**: Compiled NARMAX model simulation.
- **Constants**: `Device: str` — selected at module level via `HelperFuncs.Set_Tensortype_And_Device()`.
- **Public API**:
  - `ParsedReg2EvalStr(RegStr, VarName2Idx, NonLinName2Idx, OutputVarName) -> tuple[str, bool]` — transforms `ParsedReg` into a Python-evaluable string. Returns `(eval_str, is_autoregressive)`.
  - `VerifyParsedReg(Regressor, NonLinNames, InputVarNames, OutputVarName)` — validates variable/function declarations and causality (output lag ≥ 1). Raises `ValueError`.
  - `Make_BufferStartSystem(Expr) -> str` — rewrites Data/OutVec expressions with Toggle dispatch for buffer-start phase.
  - `Make_SystemLambdas(...) -> tuple` — generates four `eval`-compiled lambdas: `SubSystem_MA_Num`, `SubSystem_MA_Den`, `System_BufferStart`, `System_Main`. Handles rational (denominator) systems.
  - `SymbolicOscillator.__init__(InputVarNames, NonLinearities, ExprList, theta, OutputVarName, dtype, device)` — see §2.
  - `set_theta(theta)` / `get_theta()` — regression coefficient accessors.
  - `set_OutputStorage(prev) / get_OutputStorage()` — output history accessors.
  - `set_InputStorage(storage) / get_InputStorage()` — input history accessors.
  - `zeroInternalStorage()` — zeros both Input and Output storage buffers.
  - `get_nRegressors() -> int` / `get_nInputVars() -> int` — dimensionality queries.
  - `get_MaxInputLag() / get_MaxOutputLag() / get_MaxPositiveInputLag() -> int` — lag queries.
  - `Buffer_Toggle(VarNumber, k, Data) -> tor.Tensor` — selects between storage (k<0) and data (k≥0) for MA block operations across all k.
  - `Scalar_Toggle(DataOrOutVec, VarNumber, k, Data) -> tor.Tensor` — scalar version for buffer-start loop.
  - `Oscillate(Data, theta?, DsData?) -> tor.Tensor` — core simulation. Computes MA blocks on GPU, then iterates buffer-start and main loops on CPU. Updates internal state. Supports optional `DsData` (direct injection) and `theta` override. Returns `OutVec` on configured device.
    - **Errors**: `ValueError` for bad input types/shapes, dimension mismatches.

#### `NARMAX/Classes/NonLinearity.py`
- **Responsibility**: Nonlinear function container with validation.
- **Public API**:
  - `__init__(Name, f, fPrime?, fSecond?, ToMorph?)` — validates callable, output shape, NaN-free, elementwise. Raises `ValueError`/`AssertionError` on validation failure.
  - `get_f()`, `get_fPrime()`, `get_fSecond()` — return stored callables.
  - `get_Name() -> str` — return name.
  - `to_Morph() -> bool` — return morph eligibility flag.

#### `NARMAX/Classes/Morpher.py`
- **Responsibility**: Continuous optimization of morphed regressors.
- **Public API**:
  - `DMFunc(yO, Xl, ksi, PA, f) -> float` — correlation cost (orthogonalized projection).
  - `DMGrad(yO, Xl, ksi, PA, f, fPrime) -> tor.Tensor` — gradient of correlation cost.
  - `DMHessian(yO, Xl, ksi, PA, f, fPrime, fSecond) -> tor.Tensor` — Hessian (4-term decomposition).
  - `Expressionparser(idx, T) -> tuple[int, int]` — maps flat index to (monomial-index, function-index) via lookup table `T`.
  - `IsMorphable(f, x0, order) -> bool` — relaxed homogeneity test. Identity excluded; order>0 always morphable.
  - `GenVSGen(f, U, i0, Dc, order, Psi, Psi_n, y, m) -> tuple[list, list]` — brute-force vector-space generation over coefficient combinations (linspace grid). Returns selected indices and coefficients.
  - `InfOPT(y, Xl, ksiS, Ds, Dc, PA, f, fPrime, fSecond, Reps, A_T, Psi, Psi_n, W_T, L) -> tor.Tensor` — scipy `trust-exact` refinement with random restarts. Returns optimized coefficients.
  - `DictionaryMorpher(U, ell, Psi, Psi_n, y, A_T, W_T, L, Ds, Dc, MDict) -> tuple | None` — end-to-end morphing: parse → test → GenVSGen → InfOPT → metadata. Returns `(morph_data, regressor, regressor_name)` or `None` if unmorphable.

#### `NARMAX/Classes/Parser_0_3.py`
- **Responsibility**: Expression string parser for NARMAX symbolic expressions.
- **Public API**:
  - `CleanExpression(InputExpr: str) -> tuple[Optional[str], str]` — normalizes whitespace, protects lag brackets, handles `~/` and `1/` fraction operators, splits function name and arguments. Raises `ValueError` for fractional lags, reserved sequences, multi-paren.
  - `ExpressionParser(InputExpr: str) -> ParsedReg` — full parse: splits arguments, recognizes `+`, `-`, `*`, `/`, `^`, extracts coefficient/variable/lag/exponent via regex. Returns structured `ParsedReg`.
  - `DebugExpression(InputExpr: str)` — pretty-prints parse result.
- **Errors**: `ValueError` for malformed expressions, mismatched operators/subexpressions.

#### `NARMAX/Classes/Queue.py`
- **Responsibility**: Typed integer-array FIFO queue.
- **Public API**:
  - `Queue.__init__()` — initializes `deque`.
  - `put(item: NDArray[np.int64])` — enqueue. Raises `TypeError` for non-integer arrays.
  - `get() -> NDArray[np.int64]` — dequeue. Raises `QueueEmptyError` if empty.
  - `is_empty() -> bool`, `clear()`, `size() -> int`, `peek() -> NDArray[np.int64]`.

#### `NARMAX/Classes/MultiKeyHashTable.py`
- **Responsibility**: Ordered-subset lookup table (original dict-based).
- **Public API**: `__getitem__`, `SameStart`, `AddData`, `CreateKeys`, `DeleteAllOfSize`. See §2.

#### `NARMAX/Classes/New MultiKeyHashMap.py`
- **Responsibility**: Performance-optimized drop-in replacement using open-addressing hash tables.
- **Public API**: Same as `MultiKeyHashTable` above. Internal: `_commutative_hash`, `_verify_match`, `_insert`, `_resize` (2× capacity at 75% load factor).

### `NARMAX/CTors/` (constructor functions for regressor dictionary)

#### `NARMAX/CTors/__init__.py`
- **Responsibility**: Regressor dictionary construction — lagging, monomial expansion, nonlinear transforms, rational extension, boolean operations.
- **Public API**:
  - `Lagger(Data, Lags, RegNames?) -> tuple[Optional[tor.Tensor], tor.Tensor, NDArray]` — creates delayed copies. Extracts `y[k]` if `"y"` in RegNames. Returns `(y, RegMat, OutNames)`. Validates lengths, finite values, lag bounds. Raises `ValueError`/`AssertionError`.
  - `Expander(Data, RegNames, ExpansionOrder, IteractionOnly=False) -> tuple[tor.Tensor, NDArray]` — monomial expansion to given order. Uses combinations/combinations_with_replacement. Pre-allocates output matrix. Returns expanded regressor matrix and names.
    - **Parameters**: `IteractionOnly=True` generates only cross-terms (no powers).
    - **Errors**: `ValueError` for bad order/dimensions.
  - `NonLinearizer(y?, Data, RegNames, Functions, MakeRational?) -> tuple[tor.Tensor, NDArray, list[int]]` — applies elementwise nonlinear functions (skipping identity). Optionally generates rational terms (`-y * f(reg)`). Returns transformed matrix, names, and morphing index map `M`.
    - **Errors**: `AssertionError` for bad types, missing identity as first function.
  - `Booler(Data, RegNames, Operations?, OperationNames?, AllowNegation?) -> tuple[tor.Tensor, NDArray]` — boolean binary operation expansion. Pre-allocates full output matrix, uses `out=` parameter. Removes constant columns and deduplicates.
    - **Operations**: default `[and, xor, or]`. Generates all pairwise combinations with optional negation.
    - **Errors**: `ValueError` for invalid operations, shape/dtype mismatches.

### `NARMAX/Tools/` (analysis tools)

#### `NARMAX/Tools/__init__.py`
- **Responsibility**: IIR analysis and variable selection utilities.
- **Public API**:
  - `rFOrLSR2IIR(theta, L, RegNames) -> tuple[np.ndarray, np.ndarray]` — converts rFOrLSR output to (b, a) IIR coefficients. Expects `x[k-j]`/`y[k-j]` naming. Raises `ValueError` for unrecognized terms.
  - `IIR_Spectrum(b_a_List?, h_List?, FilterNames?, Fs, Resolution, xLims?, yLimMag?) -> tuple[Figure, Axes]` — magnitude/phase frequency response plot using `scipy.signal.freqz`. Accepts either (b,a) tuples or precomputed complex responses.
  - `zPlanePlot(b, a=1, Title?) -> tuple[ndarray, ndarray, float]` — pole-zero plot with unit circle. Returns (zeros, poles, gain). Auto-scales axes.
  - `ComputeERR(y, Ds) -> NDArray` — computes ERR for imposed regressors only (no candidate search). Early exits when cumul ERR ≥ 1.
  - `ExpansionOrderEstimator(x, y, MaxLags, MaxOrder, VarianceAcceptThreshold, Plot) -> tuple[int, NDArray]` — determines minimum monomial expansion order for NARX models using ERR-threshold search.
  - `MaxLagsEstimator(x, y, ModelOrder, MaxLags, VarianceAcceptThreshold, Plot, SaveFig?) -> tuple[dict, NDArray]` — computes 2-D lag grid of explained variance and recommends optimal (n_b, n_a). Outputs `Recommendations` dict with `Min_XY`, `Min_X`, `Min_Y` keys.

### `NARMAX/TestSystems/` (benchmark systems)

#### `NARMAX/TestSystems/__init__.py`
- **Responsibility**: Ground-truth NARMAX system generators for testing.
- **All functions follow same pattern**: `(x, W?, Print?) -> (x_cut, y_cut, W_cut)`; MISO/MIMO variants return per-input tensors. All center input, run a loop over `k >= MaxLag`, support additive noise.
- **Public API**:
  - `InputCheck(x, W, Print)` — validates tensor type, 1D, optional W dim match, Print bool.
  - `MA(x, W?, Print?)` — `y[k] = 0.2*x[k] + 0.7*x[k-1] -0.5*x[k-1]^2 + ...`. MaxLag=4.
  - `ARX(x, W?, Print?)` — linear ARX: `y[k] = 0.2*x[k] + 0.7*x[k-1] + ... -0.5*y[k-1] - ...`. MaxLag=4.
  - `TermCombinations(x, W?, Print?)` — mixed term combos: `y[k] = -0.2x[k] -0.4*x[k-1]^3 + 0.3*x[k-2]*x[k-1]^2 + ...`. MaxLag=3.
  - `iFOrLSR(x, W?, Print?)` — iFOrLSR paper benchmark: `y[k] = 0.2*y[k-1]^3 + 0.7*y[k-1]*x[k-1] + ...`. MaxLag=2.
  - `NonLinearities(x, W?, Print?)` — rich nonlinear benchmark with abs, exp, cos: MaxLag=3.
  - `SevereNonLinearities(x, W?, Print?)` — severe nonlinear: cos³(tan(abs(tanh(...)))). MaxLag=3.
  - `RatNonLinSystem(x, W?, Print?)` — rational system with abs in numerator and denominator. MaxLag=2.
  - `RationalNARMAX(x, W?, Print?)` — second rational benchmark with cos denominator. MaxLag=3.
  - `ThreeInputMISO(x1, x2, x3, W?, Print?)` — 3-input MISO with crossed terms. MaxLag=3.
  - `ThreeInputMIMO(x1, x2, x3, W?, Print?)` — 3-input 2-output MIMO. MaxLag=3.
  - `Binary_MISO_System(x1, x2, x3, x4, W?, Print?)` — boolean-input system using AND, XOR, OR, NOT. MaxLag=2. Requires binary inputs.
