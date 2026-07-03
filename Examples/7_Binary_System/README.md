<div align="justify">

# Example 7: Binary System Fitting & MISO (Multiple-Input Single-Output) Systems

This example / tutorial illustrates:
- The *Booler* regressor constructor: creating boolean logic combinations from binary regressors
- Fitting MISO (Multiple-Input Single-Output) systems
- One more example of creating a custom validation function using a relative MAE (Mean Absolute Error) metric
- Interpreting results when multiple equivalent expressions exist for a binary system

<br/>

# 1. The Booler

Previous tutorials used the *Lagger*, *Expander* and *NonLinearizer* CTors to create the more classical NARMAX candidate regressor dictionaries $D_C$ containing delayed terms, monomial expansions and nonlinearities. Here we pay tribute to the general NARMAX framework by also demonstrating the fitting of complex multi-input binary systems: The *Booler* CTor generates all non-redundant boolean combinations of the passed regressors.

Given a set of binary regressors $x_1, x_2, \dots, x_n$, the *Booler* generates:

1. **The original regressors** — $x_1, x_2, \dots$
2. **Their negations** (if `AllowNegation=True`) — $\ !x_1,\ !x_2,\ \dots$
3. **All pairwise combinations** using the provided operations — for each operation $\oplus \in \{\land, \oplus, \lor\}$ and each pair $(x_i, x_j)$:
   - $x_i \oplus x_j$
   - $\ !x_i \oplus x_j$
   - $x_i \oplus\ !x_j$
   - $\ !x_i \oplus\ !x_j$

Constant columns (all-True or all-False such as $x_1[k] \land !x_1[k]$) are automatically removed, as are duplicate columns (such as $!(x_1 \land x_2) \iff !x_1 \lor !x_2$), keeping the dictionary compact for faster fitting. See section 6 below for more explanations on how this might yield different looking but equivalent systems.

Default operations are `AND` (&&), `XOR` (^) and `OR` (||), but custom boolean operations can be passed via the `Operations` and `OperationNames` arguments:

```py
Operations = [tor.logical_and, torch.logical_xor, torch.logical_or]
OperationNames = ["&&", "^", "||"]
```

# 2. Training Data

## 2.1 System Under Investigation

The system is a MISO (**M**ultiple **I**nput **S**ingle **O**utput) binary system with four boolean inputs $x_1, x_2, x_3, x_4$ and a single boolean output $y$::

```math
\begin{aligned}
y[k] =\ &(!x_1[k] \land x_2[k]) - (x_3[k] \oplus x_4[k]) + (x_1[k] \lor x_3[k-1]) \\
        &+ (x_2[k] \oplus x_4[k-2]) - (!x_1[k-2] \land x_2[k]) \\
        &- !x_3[k-2] + !x_2[k-1] + x_4[k-1]
\end{aligned}
```

**Note (Binary equivalence):** Since the inputs are binary (0 or 1), many equivalent expressions exist. For example $!x \iff 1 - x$, $!(x_1 \land x_2) \iff !x_1 \lor !x_2$, and $x_1 \oplus !x_2 \iff !(x_1 \oplus x_2) \iff !x_1 \oplus x_2$. This creates a large set of zero-error equivalent systems, which is why the arborescence may not retrieve the exact original expression but rather an equally valid equivalent.

## 2.2 Input Generation

Four random binary sequences are generated using `tor.randint(0, 2, ...)` with dtype `tor.bool`:

```py
x1 = tor.randint( 0, 2, size = ( p, ), device = NARMAX.Device, dtype = tor.bool )
x2 = tor.randint( 0, 2, size = ( p, ), device = NARMAX.Device, dtype = tor.bool )
x3 = tor.randint( 0, 2, size = ( p, ), device = NARMAX.Device, dtype = tor.bool )
x4 = tor.randint( 0, 2, size = ( p, ), device = NARMAX.Device, dtype = tor.bool )

x1, x2, x3, x4, y, W = Test_Systems.Binary_MISO_System( x1, x2, x3, x4, W, Print = False )
```

## 2.3 Candidate Dictionary Creation

Like in the previous examples, the *Lagger* CTor creates the delayed copies of the input and output variables. Here, each input $x_1$–$x_4$ has a maximum lag of 2 and the output $y$ has no lag added:

```py
VarNames = [ "x1", "x2", "x3", "x4", "y" ]
Lags =     [    2,    2,    2,   2,    0 ]
y, RegMat, RegNames = NARMAX.CTors.Lagger( [ x1, x2, x3, x4, y ], Lags, VarNames)
```

The *Booler* CTor then generates all boolean combinations from the delayed regressors:

```py
RegMat, RegNames = NARMAX.CTors.Booler( RegMat, RegNames )
```

This fills the candidate dictionary $D_C$ with delayed variables, their negations, and all pair-wise boolean operations, everything needed to describe the system under investigation.

<br/>

# 3. Validation

## 3.1 Validation Data

Five independent validation sequences are generated. The validation dictionary stores the raw validation data (not yet CTor-processed) since the validation function will reconstruct the regressors on the fly (see 3.2 below):

```py
ValidationDict = { "y": [], "Data": [], "InputVarNames": VarNames, "Lags": Lags }

for i in range(5):
    x1_val = tor.randint( 0, 2, size = ( p//2, ), device = NARMAX.Device, dtype = tor.bool )
    # ... (same for x2_val, x3_val, x4_val)
    _, _, _, _, y_val, W = NARMAX.TestSystems.Binary_MISO_System( x1_val, x2_val, x3_val, x4_val, W, Print = False )
    ValidationDict[ "y" ].append( y_val.to( tor.float64 ) )
    ValidationDict[ "Data" ].append( [ x1_val.to( tor.float64 ), x2_val.to( tor.float64 ), x3_val.to( tor.float64 ), x4_val.to( tor.float64 ) ] )
```

## 3.2 Custom Validation Function

Since the candidate dictionary $D_C$ is created via the *Booler* CTor (not the standard *Expander*/*NonLinearizer* pipeline), the default `NARMAX.DefaultValidation` won't work. A custom validation function is thus required. The arborescence passes six fixed arguments; the only one the user must handle is `ValDic`,the rest are generated internally.

The custom validation function is very similar to the default one privided b the library: it reconstructs the regressor matrix from validation data using the same *Lagger* + *Booler* pipeline, computes the model output $\underline{\hat{y}} = D_C[\mathcal{L}] \cdot \underline{\hat{\theta}}$, and returns the relative MAE normed by the mean absolute value of the target:

```python
def Bool_MAE( theta, L, ERR, RegNames, ValDic, DcFilterIdx = None ):
    Error = 0
    for i in range( len( ValDic[ "Data" ] ) ):
        _, RegMat, RegNames = NARMAX.CTors.Lagger( ValDic[ "Data" ][ i ], ValDic[ "Lags" ], ValDic[ "InputVarNames" ] )
        RegMat, _ = NARMAX.CTors.Booler( RegMat, RegNames )
        if ( DcFilterIdx is not None) : RegMat = RegMat[ :, DcFilterIdx ]
        yHat = RegMat[ :, L.astype( np.int64 ) ].to( tor.get_default_dtype() ) @ theta
        Error += tor.mean( tor.abs( ValDic[ "y" ][ i ] - yHat ) / tor.mean( tor.abs( ValDic[ "y" ][ i ] ) ) ).item()
    return Error / len( ValDic[ "Data" ] )
```

**Note:** Since the validation data dictionary `ValDic` and the validation function are user-provided to the arborescence, one could store the pre-computed validation data rather than generating it on the fly if compute is more scarce / expensive that VRAM. This would make the validation function much simpler and avoid the need to recompute the regressor matrix for each validation. Since this is a tutorial, I chose to demonstrate the more complex variant.

<br/>

# 4. Running the Arborescence

The arborescence is created and fitted like in the previous examples. Since this is a pure $D_C$ fitting problem (no imposed regressors), only `Dc`, `DcNames`, the tolerances and the validation function/dictionary are passed:

```python
Arbo = NARMAX.Arborescence( y.to( tor.get_default_dtype() ),
                            Dc = RegMat.to( tor.get_default_dtype() ), DcNames = RegNames,
                            tolRoot = tol, tolRest = tol,
                            MaxDepth = ArboDepth,
                            ValFunc = Bool_MAE, ValData = ValidationDict
                          )
theta, L, ERR, _, RegMat, RegNames = Arbo.fit()
```

<br/>

# 5. Results

## 5.1 Console Output

The arborescence traverses the search space, quickly finding an 8-regressor model:

```
Performing root regression
Shortest encountered sequence (root): 8 

Arborescence Level 1: 100%|##########| 8/8 [00:00<00:00, 170.20 rFOrLSR/s]
Shortest encountered sequence: 8

Arborescence Level 2: 100%|##########| 56/56 [00:00<00:00, 226.63 rFOrLSR/s]
Shortest encountered sequence: 8

Arborescence Level 3: 100%|##########| 336/336 [00:00<00:00, 374.10 rFOrLSR/s]
Finished Arborescence traversal.
Shortest encountered sequence: 8

Starting Validation procedure.
Validating: 100%|##########| 139/139 [01:02<00:00,  2.23 Regressions/s]

Validation done on 128 different Regressions. Best validation error: 2.4863318417198824e-17
```

Out of 401 possible regressions, only 208 were actually computed (51.9%), of which 69 were early-aborted via the OOIT (Orthogonalization Order Independece Theorem).  
The best validation error is about $2.5 \times 10^{-17}$ — essentially floating point noise — confirming that the found expression is an exact equivalent of the original system:

```
AOrLSR Results:
-1.0 x2[k-1]
 1.0 !x1[k] && x3[k-1]
 1.0 x1[k] && !x2[k]
 2.0 x3[k-2] && x4[k-1]
 1.0 x2[k] ^ x4[k-2]
 1.0 !x3[k] ^ x4[k]
 1.0 x3[k-2] ^ x4[k-1]
-1.0 !x1[k-2] || !x2[k]
```

## 5.2 Residual Plot

![Figure 1](https://github.com/Stee-T/NARMAX/blob/main/Examples/7_Binary_System/Figure_1.png)

The residual plot shows the difference $y[k] - \hat{y}[k]$. With a validation MAE of $\sim \!10^{-17}$, the residuals are at the level of floating-point rounding errors, confirming that the fitted model is mathematically equivalent to the original system, even if the equation might be differnt (see next seciton for explanations).

<br/>

# 6. Discussion: Equivalent Binary Expressions

Binary systems present a unique challenge for symbolic regression: the restricted input set (only 0 and 1) means many different expressions produce identical outputs. As discussed in the paper:

> *"This system is hard to find, since not only the input set is restricted to binary values, which reduces the algorithms' comparison possibilities, but there are also many equivalent binary expressions such as $!(x_1[k] \land x_2[k]) \iff !x_1[k] \lor !x_2[k]$ or $x_1[k] \oplus !x_2[k] \iff !(x_1[k] \oplus x_2[k]) \iff !x_1[k] \oplus x_2[k]$, etc., which further reduces the probability to find the exact original equation. The set of correct systems is even larger than only equivalent regressors, since the regressions can use arbitrary regression coefficients $\hat{\underline{\theta}}$ to add or subtract multiples of each regressor, which opens up more possibilities for equivalent, thus zero-error systems."*

This is exactly what we observe: the arborescence finds an 8-term equation that is not the original but achieves essentially zero error on unseen validation data — a perfect fit from an equivalent expression.

<br/>

[Previous Tutorial: 6. Multiple Input Systems](https://github.com/Stee-T/NARMAX/tree/main/Examples/6_MIMO)  
[Next Tutorial: 8. Derivative ODE Discovery](https://github.com/Stee-T/NARMAX/tree/main/Examples/8_Derivative_ODE_Discovery)

</div>
