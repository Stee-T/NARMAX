<div align="justify">

# Example 2: Supplementary infos & Rational System
This example / tutorial illustrates:
- How to create reproductible arborescence runs.
- How to use the NonLinearizer to create rational expansions.
- Why I proposed the arborescence as a (r)FORLSR amelioration.
- Some more info an how to interpret the console output

<br/>

# 1. Reproducibility
This code segment illustrates how to create reproductible arborescence / rFOrLSR runs for testing and comparison purposes.

The AOrLSR (and FOrLSR) is a fully deterministic algorithm, not having any intrinsic stochastic processing. If the (randomly generated) input sequence $\underline{x}$, the system output $\underline{y}$ and the error sequence $\underline{e}$ and all hyper-parameters, thus both dictionaries' $D_C$ and $D_S$ content remain the same, the arborescence will generate the same results.

The below code-snippet demonstrates how to set a seed and reliably generate a random sequence $\underline{x}$ of length $p$. Here NumPy is used, since it provides more equality guarantees than the random number generator of PyTorch (at the time of writing and to the best knowledge of the author). NumPy guarantees that the same seed will generate the same sequence across any operating system and any hardware, which is not the case for the PyTorch random number generator. 

As always, it's very important to center the input sequence before sending it through the system-under-investigation.

```python
import numpy as np
Seed = 252 # use a fixed seed to reproduce results
print ( "\nSeed: ", Seed, "\n" )
RNG = np.random.default_rng( seed = Seed )
x = tor.tensor( ( 2 * InputAmplitude ) * ( RNG.random( p ) - 0.5 ) ) # uniformly distributed white noise
x -= tor.mean( x ) # center
x, y, W = Sys( x, W, Print = True ) # apply selected system
if ( tor.isnan( tor.sum( y ) ) ): raise AssertionError( "Yields NaNs, which we don't like" )
```
<br/>

# 2. Constructing Denominator Regressors
This code snippets demonstrates how to construct denominator regressors with only the absolute value as non-linearity.

The handling of non-linearities was explained in the [first example / tutorial](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/1_Linear_in_the_Parameters "Basics and Linear-in-the-parameter system fitting Example"). The only addition is the Boolean iterable `MakeRational` containing a True / False for each non-linearity determining if it is to be made into a denominator regressor. This illustrates one of the reasons why the `NonLinearities` list needs to start with an Identity (object), as unprocessed regressors must have the option to be put in the denominator.

Trivially, the `MakeRational` argument needs to be passed to the *NonLinearizer* signal-CTor and to the validation function to be applied during test and validation signal construction. 

```python
NonLinearities = [ rFOrLSR.Identity ] # List of NonLinearity objects, must start with identity
MakeRational = [ True ] # List of bool, deciding if each NonLinearity is to also be made rational

NonLinearities.append( rFOrLSR.NonLinearity( "abs", f = tor.abs ) )
MakeRational.append( True )
```

For code clarity, I recommend appending the bool to `MakeRational` under each addition to the `NonLinearities` list to link both entries.

<br/>

# 3. Learnings Of Arborescence For Make Benefit Glorious rFOrLSR
As illustrated in the below print-out, the example system is so complex that the (r)FOrLSR performs very poorly on its own. "On its own" means that no further arborescence is spanned by imposing regressors to form new arborescence branches. Thus, the rFOrLSR "on its own" is an arborescence of depth zero (root only).

The printout displays that the number of regressors used to describe the system drops at each traversed level until the correct system equation is retrieved.

**Level 0 (root):** The FOrLSR "on its own", being the root requires 29 regressors to meet the desired precision threshold (`tolRoot` $=\rho_1$), which is a very poor approximation, being more 3.6 times the correct amount of regressors. Essentially, each time a wrong regressor is chosen, a few supplementary are needed to compensate for the error. Models with many regressors are generally overfitted and don't generalize well to unseen data.

**Level 1:** After the first arborescence level, the precision threshold (`tolRest` $=\rho_2$) is already met with 13 regressors, which is already half the previous estimation. This was achieved by imposing one of the regressors chosen by the root at the regression start, then letting the rFOrLSR choose the following regressors freely. This forces the rFOrLSR to start in different search-space locations to hopefully find better solutions.   
In this example, the root has 29 children nodes, meaning that 29 regressions will be computed at level 1 with each one getting a different regressor from the root sequence imposed at the start. 

**Remaining Levels:** Level 2 finds a sequence with 9 regressors, further reducing the best-known solution by 4 regressors, while Level 3 just adds same length regressor sequences for the validation to choose from. Level 4, however, finds the correct solution.  
It's not uncommon to have a few levels in a row not decreasing the regressor sequence length followed by one or more levels decreasing it. It is thus not a good heuristic to stop the arborescence after a level not decreasing the regressor sequence length. 

**Fitting Stability:** In addition to finding better solutions at each level,  the arborescence is also a lot more stable than the FOrLSR on itself. Indeed, for this system, the arborescence always finds the correct solution after worst case 5 levels, while the rFOrLSR (root only) hasn't found it in any performed simulation. See the paper for more [#TODO Link] details and the benchmarks. I thus recommend to always let the arborescence span at least 3 levels, which is very quick on even the weakest Laptops.

```
Seed:  252

System: y[k] = ( 0.6*abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*abs( y[k-1] )  ) / ( 1 - 0.4*abs( x[k] ) + 0.3*abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] )

Performing root regression
Shortest encountered sequence (root): 29

Arborescence Level 1: 100%|████████████████████████| 29/29 [00:01<00:00, 19.24 rFOrLSR/s]
Shortest encountered sequence: 13

Arborescence Level 2: 100%|██████████████████████| 670/670 [00:19<00:00, 38.16 rFOrLSR/s]
Shortest encountered sequence: 9

Arborescence Level 3: 100%|██████████████████| 14265/14265 [03:15<00:00, 86.25 rFOrLSR/s]
Shortest encountered sequence: 9

Arborescence Level 4: 100%|███████████████| 280983/280983 [20:09<00:00, 287.43 rFOrLSR/s]
Finished Arborescence traversal.
Shortest encountered sequence: 8

Starting Validation procedure.
Validating: 100%|████████████████████| 37349/37349 [00:00<00:00, 765434.50 Regressions/s]

Validation done on 1 different Regressions. Best validation error: 1.376945153063429e-15
Out of 295948 only 70642 regressions were computed, of which 33293 were OOIT - aborted.

8 Terms yielding an Mean absolute Error (MAE) of 1.013e-14 % and a maximal deviation of 1.164e-13% and a Median Absolute Deviation (MAD) of 8.3627e-15
```
**Note 1 (Displayed speed):** The displayed fitting times at the end of each progress-bar aren't the level average but the last prediction the progress-bar made and is thus not representative of the overall fitting time.  
It is, however, almost always the case that the average number of regression computed per second drop with higher levels, since more regressions are known and thus not computed at each level.

**Note 2 (Example speed):** Don't be scared by the processing times, this was done on my 9.5-year-old laptop :). Modern GPUs have much faster processing times, especially industrial ones.

**Note 3 (Denominator display):** The denominator regressor (those below the fraction) are denoted by ~/(regressor) to represent $\frac{\tilde{}}{\text{regressor}}$. This notation (where ~ is a placeholder) avoids confusing those denominator regressors with actual fractions, as regressors of the form $\frac{1}{\text{regressor}}$ are also valid.

```
Recognized regressors:
 0.5999999999999982 abs(x[k])
-0.3499999999999991 x[k]^3
-0.2999999999999997 y[k-2] x[k-1]
 0.2999999999999996 ~/(y[k-1] x[k-2])
-0.40000000000000263 ~/(abs(x[k]))
 0.3000000000000012 ~/(abs(x[k] x[k-1]))
-0.20000000000000034 ~/(x[k-1]^3)
 0.10000000000000045 abs(y[k-1])
```
