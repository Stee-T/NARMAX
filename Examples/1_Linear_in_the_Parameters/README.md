<div align="justify">

# Example 1: Basics & Linear-in-the-parameter System

This example / tutorial illustrates
- Library basics: imports, hyper-parameters and their meaning
- Signal constructors: *Lagger*, *Expander*, *NonLinearizer*
- Using the Default validation function and creating the required data and hyper-parameter dictionary
- Running the arborescence
- Interpreting the results


## 1. Imports
The main dependency is PyTorch which is used to do matrix operations on the GPU. Make sure that CUDA or MPS (M1/M2 Macs) is installed and that your PyTorch installation recognizes it. The library will issue a warning if no GPU is usable, as the fitting will be much slower.

A list of dependencies is available on [the main git page](https://github.com/Stee-T/rFOrLSR/tree/main).  
We also tell matplotlib to use a dark background, because it's no longer the 90s and we don't like looking at the sun.  
The rFOrLSR.Test_Systems contains many systems to demonstrate the library. In real word usage, the system is the system under investigation.


```python 
import torch as tor

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import rFOrLSR_0_14 as rFOrLSR
import rFOrLSR_0_14.Test as Test_Systems
```

## 2. Hyper-parameters
Once the imports are done, some arborescence and signal-constructor hyperparameters must be chosen. This example uses the following parameters:

**Dataset-size p:** TL;DR: More data = Better results, longer fitting times  
Dictates the regressor (vector) length and influences linearily the fitting time. Good values depend on the system complexity, as for small linear systems, $p = 500$ often suffises, however for more complicated systems, larger values like $p = 2.5k$ or higher could become necessary. If your GPU is fresh, go for $p = 100k$, whatever really.

**Signal amplitude:** TL;DR: More Input-signal $x$ amplitude/variance = better regressor recognition.  
The rFOrLSR describes system behavior with the regressors passed via $D_C$, so the more "input space" is covered by the excitation signal $\underline{x}$, the better the rFOrLSR sees if a function is matching.
The input amplitude has no influence on fitting time. In general, it should be chosen as large as possible (before the system saturates or becomes unstable) and should at least cover the range in which the system will be used.  
To illustrate, for $\underline{x}\in[-0.4;0.4]^p$, $\cos(x)$ and $-0.5x^2+1$ are almost the same, while for $\underline{x}\in[-2.5;2.5]^p$ those functions look very different.

**Tolerances $\rho_1, \rho_2$:** TL;DR: tolerance going towards zero = more precision, larger models, longer fitting times.  
The arborescence takes two tolerances as input: $\rho_1$, being the root regression tolerance and $\rho_2$, being the tolerance for every other regression / arborescence node.  
In simple terms, those thresholds represent what fraction (thus $[0; 1]$) of the system output $\underline{y}$'s variance remains unexplained by the model. Thus a tolerance of 0.1 means that 10% of the variance is unexplained by the model, a tolerance of 0.01 is 1%, etc. The computations are based on some unintuitive square correlation ratios.  
As will be explained below, the arborescence fitting report contains more intuitive fitting statistics such as the MAE (mean absolute error), maximal deviation (MD), etc.  
Using a different tolerance for the root (mostly closer to zero than desired for the model) allows to span a wider (more breadth) arborescence. Having a lower residual variance tolerance, the root node will use more regressors and thus generate more children nodes which will become branches.

**Maximum delay $q_x$, $q_y$:** Those are the maximum lag arguments passed to the *Lagger* signal constructor to obtain delayed regressors for temporal system fitting. Those can of course later be passed through non-linearities and combined with spatial regressors to create spatio-temporal regressors.  
To illustrate, in this example $q_x=4$ means that from the passed signal $\underline{x}$, the regressors $\underline{x}\[k\],\ \underline{x}\[k-1\],\ \underline{x}\[k-2\],\ \underline{x}\[k-3\],\ \underline{x}\[k-4\]$ will be constructed. Obviously, for the auto-regressive terms ($\underline{y}\[k-j\], j\in \mathbb{N}$) the lags start at $j=1$, ($y\[k\]$ is the regressor to end all regressions) and ends at $j=q_y$.

**ExpansionOrder:** Monomial expansion order for the *Expander* signal-CTor determining the upper bound of the sum of exponents of each generated term.  
To illustrate, be a set of two regressors with 3rd order expansion: $\[\ \underline{x}_1,\ \underline{x}_2\ \]→\[\ \underline{x}_1,\ \underline{x}_2,\ \underline{x}_1^2,\ \underline{x}_2^2,\ \underline{x}_1\cdot\underline{x}_2,\ \underline{x}_1^3,\ \underline{x}_2^3,\ \underline{x}_1^2\cdot\underline{x}_2,\ \underline{x}_1\cdot\underline{x}_2^2\ \]$
$\underline{x}_1$ is a correct expansion term, since $\underline{x}_1=\underline{x}_1^1\cdot \underline{x}_2^0$ and $1 + 0 \leq 3$. In general any monomial of exponent $n \leq$ ExpansionOrder is valid: $\underline{x}_a^n = \underline{x}_a^n \cdot \prod_{j\in J} \underline{x}_j^0$, as $n+\Sigma_{j\in J}0=n$.
Similarly, in this example $\underline{x}_1^2\cdot\underline{x}_2$, is valid since $2+1\leq3$.

**ArboDepth:** TL;DR: More levels = shorter models, lower validation errors, longer fitting times.  
Maximum number of levels the arborescence can have.  
The arborescence spans it levels by imposing progressively more regressors at the start of the regression which forces the rFOrLSR to explore more of the solution space.  
To illustrate, At the root (level 0), no regressors are imposed, resulting in a fully free rFOrLSR search. At the first level, one tree branch is created for each regrerssor selected at the root to force the rFOrLSR to start with that regressor. At the second level, two regressors are taken from the parent node and imposed at the start, etc., etc.  
**Note**: Following my *Arborescence Depth Truncation Theorem (ADTT)*, the `ArboDepth` parameter is a non-tight upper-bound, as the arborescence can judge that a smaller number of levels suffices. This is the case if (and only if) a model with fewer regressors than `ArboDepth` was encountered during the traversal, since then any longer model will be rejected by the validation procedure and thus doesn't need to be computed.


```python 
p = 2_500 # Dataset size
InputAmplitude = 1
tol = 0.0001 # Fitting tolerance in %, so multiply by 100
W = None # No Noise in this example
qx = 4; qy = 4 # maximum x and y delays
ExpansionOrder = 3 # monomial expansion order
ArboDepth = 4 # maximum number of levels the arborescence can have
```
Once the main hyper-parameters are selected, one needs to create the input sequence $\underline{x}$ and the output sequence $\underline{y}$. In this example, $\underline{y}$ is generated by one of the example systems provided by the library for demonstration (feel free to uncomment the desired system). In real word usage, $\underline{y}$ is obviously the system / function to be fitted.  

**Very important:** The Input sequence $\underline{x}$ must be centered / zero-mean before going into the system and before being processed by the provided signal-CTors. This is required for the rFOrLSR to correctly fit the system. The library takes care of all other required centering, so no other particular care needs to be taken. However, this centering can't be automated since it affects the processing performed by the system / function under investigation.

The while loop represents that we'd like to avoid NaNs in the output $\underline{y}$, which can happen for certain $\underline{x}$ sequences on the given Test systems.

rFOrLSR.device is a string conaining the device (CPU / GPU) currently used by the library.

```python 
# System choice
# Sys = Test_Systems.ARX # IIR filter
# Sys = Test_Systems.TermCombinations
# Sys = Test_Systems.iFOrLSR
Sys = Test_Systems.NonLinearities # Example 1 in the paper
# Sys = Test_Systems.NotInDictionaty

# Generate x and y data
while ( 5 ): # 5 is the absolute truth
  x = InputAmplitude * ( tor.rand( p, device = rFOrLSR.device ) * 2 - 1 ) # uniformly distributed white noise
  x -= tor.mean( x ) # center: VERY IMPORTANT!
  x, y, W = Sys( x, W, Print = True ) # apply selected system
  if ( not tor.isnan( tor.sum( y ) ) ): break
```

**Non-linearities:** If desired, the NonLinearizer signal-CTor can be used to add non-linearities to the candidate regressor dictionary $D_C$.  
Non-Linearities must be declared by adding them to a list starting with the provided identity as `rFOrLSR.NonLinearity` objects. These objects take the non-linearity name and the function-pointer to be applied. Many security checks are performed in their constructor to assure that the function is usable for fitting, as the user can pass arbitrary function pointers.  
Note that here only the name and the non-linearity itself are passed, since no morphing is performed. The morphing also requires the first and second derivatives of each function.  
User-defined function-pointers are expected to point towards functions accepting torch tensors and returning torch tensors of the same dimensions.  
Importantly, the NonLinearities list must start with the identity, to allow the creation of rational regressors and for the morphing.

```python
NonLinearities = [ rFOrLSR.Identity ] # List of NonLinearity objects, must start with identity
NonLinearities.append( rFOrLSR.NonLinearity( "abs", f = tor.abs ) )
NonLinearities.append( rFOrLSR.NonLinearity( "cos", f = tor.cos ) )
NonLinearities.append( rFOrLSR.NonLinearity( "exp", f = tor.exp ) )

```
<br/>

## 3. Training Data (Candidate Regressor Dictionary $D_C$) Creation
Once all hyper-parameters are set, the provided signal-CTors are used to create the candidate regressor dictionary $D_C$. Each signal-CTor of course also updates the regressor names. The *NonLinearizer* outputs a 3rd argument, which is for the morphing procedure, it will thus be explained in later tutorials.

```python 
y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x, y ), MaxLags = ( qx, qy ) ) # Create the delayed signal terms
RegMat, RegNames = rFOrLSR.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = rFOrLSR.CTors.NonLinearizer( y, RegMat, RegNames, NonLinearities) # add the listed terms to the Regression matrix
```
<br/>

##  4. Validation Data
Since many models are created by the arborescence, a validation procedure chosing the best system is required. This basic example demonstrates the use of the provided `rFOrLSR.DefaultValidation` function with its required data.

**DefaultValidation:** The the provided `rFOrLSR.DefaultValidation` is designed to imitate step 3. Thus, all 3 CTors are called with their arguments contained in the `ValidationDict` dictionary. Signal-CTors are bypassed by setting their arguments to the respective identities, being (0,0) for the *Lagger*, 0 for the ExpansionOrder and the [ rFOrLSR.Identity ] for the NonLinearizer with additionally None for the *MakeRational* argument (which is explained in the next tutorial).  
The code generates 5 validation sequences to test the solutions on 5 different input sequences.

**Custom Validation Function and Data:** This will be demonstrated in a further tutorial. The arborescence expects a function pointer and a dictionary, so under mild formal conditions (fixed number of arguments, etc.), the user can use arbitrary validation procedures.

```python 
ValidationDict = { # contains essentially everything passed to the CTors to reconstruct the signal
  "y": [],
  "Data": [],
  "DsData": None, # No impopsed terms in this example
  "Lags": (qx,qy),
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": NonLinearities,
  "MakeRational": None, 
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ): # 5 is the absolute truth
    x_val = tor.rand( int( p / 3 ) ) # Store since used twice
    x_val -= tor.mean( x_val ) # center
    x_val, y_val, W = Sys( x_val, W, Print = False ) # _val to avoid overwriting the training y
    if ( not tor.isnan( tor.sum( y_val ) ) ): break # Remain in the loop until no NaN
  
  ValidationDict["y"].append( y_val )
  ValidationDict["Data"].append( ( x_val, y_val ) )
```
<br/>

## 5. Running the Arborescence
Now that the training data and the validation data are created, the arborescence can be run by first creating an `rFOrLSR.Arborescence` object then calling its `fit()` method.  

**Backups:** The code also demonstrates how to set-up regular backups via the `FileName` (saving Path with filename) and `SaveFrequency` (in minutes) arguments. Those can be omitted if no backups are required. The arborescence can be interrupted at any time by just killing the python process running it and later restarted at will. The commented-out section demonstrates how to continue the fitting after an interruption.

```python 
File = "Some/Valid/Path/FileName.pkl"

Arbo = rFOrLSR.Arborescence( y,
                             Ds = None, DsNames = None, # Ds & Regressor names, being dictionary of selected regressors
                             Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                             tolRoot = tol, tolRest = tol, # \rho tolerances
                             MaxDepth = ArboDepth, # Maximal number of levels
                             ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict, # Validation function and dictionary
                             Verbose = False, # Print current rFOrLSR state (only meaningful for regressions with many terms)
                             FileName = File, # Path and File to save the Backups into
                             SaveFrequency = 10, # Save frequency in minutes
                           )

Arbo.fit()

# If the Arborescence was interrupted and saved, continue with:
# Arbo = rFOrLSR.Arborescence() # init empty Arbo
# Arbo.load( File ) # load pickle file
# Arbo.fit() # resume fitting
```
<br/>

## 6. Analysing the results
Once the arborescence has been traversed, the results can be printed out and displayed by calling the `Arbo.PlotAndPrint()` method or obtained as vectors from the `Arbo.get_Results()` method.

**PlotAndPrint:** Generates two plots and (optionally) prints the regression results in the console.
The code also demonstrates how to force a zoom-in by setting the returned axes limits.

**get_Results / fit:** Both functions return the regression results in a tuple containing the regression parameters $\underline{ \hat{ \theta } }$, the index-set $L$ (which doesn't contain the imposed regressors) and the ERR vector. In addtion, if the arborescence was run with morphing instructions (not the case in this tutorial), a Morphing-meta Data dictionary is returned as 4th return. Finally, the updated $D_C$ and DcNames are returned as 5th and 6th return. Updating $D_C$ consists in filtering out redundant regressors (necessary for a well functioning rFOrLSR) and potentially expandin the dictionary with new morphed regressors.

**Note (callable model):** The Arborescence doesn't return an object containing a callable model, since a) the user can pass any regressor in the form of a vector (including acausal ones for dynamic systems) and b) the morphing procedure has the potential to generate arbitrary terms.

```python 
Figs, Axs = Arbo.PlotAndPrint() # returns both figures and axes for further processing, as as the zoom-in below
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

theta, L, ERR, _, RegMat, RegNames = Arbo.get_Results()

plt.show()
```

## Console Output
**Note:** Your results when running the script might differ, since the random input sequence $\underline{x}$ has some influence on the search space. However, the resulting model should be the same up to rounding errors.

First, the test systems print their regressors for ease of testing. This is followed by the usual arborescence information. We see that the root doesn't find the correct / optimal system but that the arborescence finds it during the first level. Each progress-bar displays the number of regressions to compute and the current progress in time.

**Processing times:** TL;DR: Don't trust the progress-bar predictions, most search spaces are massively redundant.  
It is impossible to predict how long a level takes since the number of computed nodes / regressions depends on how redundant the search space is which is (currently?) unpredictable. Most search spaces are, however, largely redundant (hence the AOrLSR) such that a very small percentage of the search space is actually computed.  
To illustrate, it is possible that the first 5% (containing only non-redundant regressions, which must be computed) take much longer than the remaining 95% (having only redundant regressions, which aren't computed). Thus, the progress-bar's predictions are not to be trusted and one shouldn't be demotivated by huge numbers :)

The below report shows how fast the search space is traversed (especially that this was done on my 9.5-year-old laptop with a weak GPU) thanks to the redundancy theorems. Indeed, for this input sequence $\underline{x}$ on 4 levels, only 644/4159 (15.48%) regressions were computed, of which 479 (74.37%) were early aborted.

```
System: y[k] = 0.2x[k] + 0.3x[k-1]^3 + 0.7|x[k-2]*x[k-1]^2| + 0.5exp(x[k-3]*x[k-2]) - 0.5cos(y[k-1]*x[k-2]) + 0.2|x[k-1]*y[k-2]^2| - 0.1y[k-3]^3

Performing root regression
Shortest encountered sequence (root): 10 

Arborescence Level 1: 100%|██████████████████████████████████████████| 10/10 [00:00<00:00, 30.19 rFOrLSR/s] 
Shortest encountered sequence: 7

Arborescence Level 2: 100%|██████████████████████████████████████████| 85/85 [00:01<00:00, 65.74 rFOrLSR/s] 
Shortest encountered sequence: 7

Arborescence Level 3: 100%|██████████████████████████████████████████| 590/590 [00:03<00:00, 156.58 rFOrLSR/s]
Shortest encountered sequence: 7

Arborescence Level 4: 100%|██████████████████████████████████████████| 3473/3473 [00:08<00:00, 423.79 rFOrLSR/s] 
Finished Arborescence traversal. Shortest encountered sequence: 7
```

**Validation print-out:** The validation procedure printout shows that out of all 4159 regressions (arborescence nodes) only 165 were different models (rather than permutations of each other's regressors). Also, the printout informs that only one was of the minimal length of 7 regressors, thus a single validation function-call was made. 

```
Starting Validation procedure.
Validating: 100%|██████████████████████████████████████████| 165/165 [00:00<00:00, 1815.08 Regressions/s] 

Validation done on 1 different Regressions. Best validation error: 1.244266372737191e-15
```

**Model metrics:**
Metrics based on the absolute value and the median are used rather than their classical counter-parts with squares and means (MSE and variance). This is a lot more representative, since there no squaring, which reduces numbers < 1 towards zero and enlarges numbers > 1. Thus, small errors are not ignored and outliers aren't amplified. Similarly, the median is much more robust to outliers and distribution-skewness than the mean.

The metrics are given in percentage, as the error is normed by $\max(|\underline{y}|)$ to make it independent of $\underline{y}$'s amplitude.

- **Mean Absolute Error (MAE):** $\text{MAE} := \frac{\Sigma_{k=1}^p |y[k] - \hat{y}[k]|}{p \cdot \max(|\underline{y}|)} \cdot 100$  
More representative than the MSE (mean squared error), since no squaring is performed, see above.  
This is also what the `rFOrLSR.DefaultValidation` computes. The norming is however done with $\Sigma_{k=1}^p |y[k]|$ instead of $\max(|\underline{y}|)$.

- **Maximal Deviation (MD):** $\text{MD} := \frac{\max(|\underline{y} - \underline{\hat{y}}|)}{\max(|\underline{y}|)} \cdot 100$.  
Largest error for the given input sequence $\underline{x}$.

- **Median Absolute Deviation (MAD):**  $\text{MAD}:=\text{med}(\underline{e} - \text{med(\underline{e})} )\cdot 100$  with $\underline{e} := \frac{|y[k] - \hat{y}[k]|}{\max(|\underline{y}|)}$  and "med" the median.  
Similar to the classical standard deviation, only that the absolute value is used instead of squares and the median is subtracted from each sample instead of the mean.

- **No passed argument validation metric:** If no validation data is passed to the arborescence, 1 - sum(ERR) is used as validation error for each regression. this corresponds to the remaining unexplained variance left in $\underline{y}$ by the model.

```
Out of 4159 only 644 regressions were computed, of which 479 were OOIT - aborted.

7 Terms yielding a Mean absolute Error (MAE) of 4.323e-15 % and a maximal deviation of 3.278e-14% and a Median Absolute Deviation (MAD) of 3.2780e-15
```

Finally, the model is printed out as a list of regressors and their coefficients.

```
Recognized regressors:
 0.5 exp(x[k-2] x[k-3])
 0.7 abs(x[k-1]^2 x[k-2])
-0.39999999999999997 y[k-3]^3
 0.19999999999999998 x[k]
 0.3000000000000001 x[k-1]^3
-0.49999999999999983 cos(y[k-1] x[k-2])
-0.39999999999999986 abs(y[k-2]^2 x[k-1])
```

## Figure 1
![Figure1](https://github.com/Stee-T/rFOrLSR/blob/main/Examples/1_Linear_in_the_Parameters/Figure_1.png)

**Top Graph:** Overlays the actual system response $y\[k\]$ with the estimated system response $\hat{y}\[k\]$. In the ideal case (like here) where the fitted system is correctly retrieved and there is no noise, the two curves  overlap. The estimation is on top of the system output for more visibility.

**Bottom Graph:** Displays the residuals $y\[k\] - \hat{y}\[k\]$. Remember that the top left corner can display the y-axis scaling.  
In the ideal case (like here) where the fitted system is correctly retrieved and there is no noise, the residuals should be almost zero due to rounding errors.  
This graph's title contains the same statistics as described above in the console print-out section.

**x-axis linking:** Both x-axes are linked, such that scrolling are mirrored in both graphs.

## Figure 2
**Warning / Disclaimer:** This graph is legacy from the original rFOrLSR algorithm family, where the regressor order had a meaning, as they were selected in order of how much system output $\underline{y}$ variance they explained. Now, since the arborescence is spanned by imposing regressors in a pre-determined order, the regressor order becomes meaningless. This is problematic, since the ERR is order dependent. Thus, the regressors at the end of the list (depending on the arborescence depth) might not be the one with the least importance (contribution to the system output variance). The ERR graph is thus not to be used for model truncation.  

The bottom graph shouldn't be used for model truncation either, as the model coefficients $\hat{\underline{\theta}}$ must be re-estimated each time a regressor is eliminated. This is correctly done here, but as aforementioned, this analyses only this precise regressor ordering.  
Correctly pruning a NARMAX model is non-trivial and will be covered in a further tutorial.  
Further research is needed to make those graphs contain more relevant information.

So now that I've crushed your regressor importance estimation dreams, enjoy the nice-looking graphs :)

![Figure2](https://github.com/Stee-T/rFOrLSR/blob/main/Examples/1_Linear_in_the_Parameters/Figure_2.png)

**Top Graph:** 
Contains the ERR of the regressors in the order imposed by the arborescence order (see warning above).

**Bottom Graph:** 
Contains the MAE reduction of the regressors in the order imposed by the arborescence (see warning above) with the regression coefficients re-estimated for each additional regressor.