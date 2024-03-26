<div align="justify">

# Example 6: MIMO & MISO Systems

This example / tutorial illustrates:
- Advanced use of the Lagger Regressor-CTor (Custom `MaxLags` and `RegNames` arguments)
- Handling of multiple arborescences to fit MIMO systems (Validation dictionaries, etc.)

<br/>

**On MISO systems:** The rFOrLSR naturally supports MISO (Multiple Input Single Output: $\mathbb{R}^n → \mathbb{R}$) systems / functions with multiple input channels / variables (such as $x_1, x_2,x_3,$ etc) which can be fed back into its input as $y$ term. Being based on vector matching, it doesn't remotely care about what each dictionary $D_C$ vector represents for us mere humans. The rFOrLSR can't differentiate between transformations of the same variable ($x$ vs $x^2$) or a different variable altogether ($x_1$ vs $x_2$).  
In fact, all previous examples / tutorials fit two different variables (input $x$ and output $y$), thus, strictly speaking we've been fitting MISO systems all along. In real life, however, MISO systems are considered to have distinct input variables or multiple input channels.  
To illustrate, a robot / machine with multiple sensors at locations $x_1, x_2, x_3$ will be considered a "proper" MISO system. Identically, a Japanese soup-bowl with thermometers at different locations is also a valid MISO system.

**On MIMO systems:** MIMO (Multiple Input Multiple Output: $\mathbb{R}^n → \mathbb{R}^m:n,m>1 $) system / functions are not directly supported by the rFOrLSR, however it suffices to separate each output dimension / channel into its own MISO system. Trivially, it thus suffices to run multiple arborescences on the same dictionary.

This tutorial illustrates the advanced use of the Lagger Regressor-CTor as required by MISO systems and how to handle fitting multiple output channels.

<br/>

# 1. Advanced Lagger Usage
The major difference in handling MIMO systems compared to the usual system is the handling of the multiple output vector $\underline{y}$.  
The Regressor-CTors are designed to simplify the generation of regressors for single output systems, as they handle the $\underline{y}$ vector automatically. For MIMO systems, the output vectors $\underline{y}_i$ must be handled manually.

**Lags:** First, the `Lagger` CTor is told to not add the respective $\underline{y}_i\[k\]$ vectors to the generated dictionary $D_C$. It is simpler to not have those regressors in $D_C$ and to construct them manually than to find, extract and eliminate them from $D_C$. The `Lags` argument doesn't only support integer arguments (defining the maximum lag for the regressor at that index) but also supports iterables, which provides an easy way to exclude the undesired regressors: 

``` python
Lags = ( 3, 3, 3, [1, 2, 3], [1, 2, 3] ) # x1, x2, x3, y1, y2. Exclude y1[k] and y2[k]
```

As stated in the comment, this data-structure tells the `Lagger` CTor to generate all lags from 0 up to and including 3 for the system inputs ($x_1, x_2, x_3$) but exclude the 0 lag for the outputs ($y_1, y_2$), thus not including the $\underline{y}_i\[k\], i\in \left[1,2 \right]$ vectors.


**Regressor Names:** In previous tutorials, the `Lagger` CTor was never explicitly told the regressor names. If a single regressor is passed, it assumes its name to be $x$, if two are passed it assumes $x,y$ and if three are passed it assumes $x, y, e$ ($e$ being the internal noise terms).
Here, however, since different inputs and outputs are lagged, one must explicitly tell `Lagger` the regressor names via the `RegNames` argument:

``` python
RegNames = [ 'x1', 'x2', 'x3', 'y1', 'y2' ]
```

**Order:** Importantly, the `Data`, `Lags` and `RegNames` arguments are assumed to be in the same order by the `Lagger` CTor.

**$y$ variable:** The `Lagger` looks for a generated variable called $y\[k\]$ in the dictionary $D_C$. If it is not found, it outputs None. Thus, one can't use the function output at that position for MIMO systems.

**Outputs:** The `Lagger` CTor generates lagged regressors by tactically cutting the vectors. Thus, resulting regressors are shorter than the original vectors, which must be done manually for the system outputs $\underline{y}_1, \underline{y}_2$. The library provides the convenience function `rFOrLSR.CutY`, which recursively checks for the maximum lag then returns the cut $\underline{y}$ vector:

``` python
# Cut y1, y2 to the same length as RegMat
y1 = rFOrLSR.CutY( y1, Lags )
y2 = rFOrLSR.CutY( y2, Lags )
```

<br/>

# 2. Expander and NonLinearizer
Nothing in particular to be done here. The `Expander` expands and combines all variables present in the dictionary it receives, irrespective of their names.

The `NonLinarizer` applies the passed non-linearities to the passed dictionary. Note that here "None" is passed as $\underline{y}$ as no single output exists for the system. Fit rational MIMO systems would require calling the `NonLinarizer` twice with the respective $\underline{y}$ vectors, one for each output.

``` python
NonLinearities = [ rFOrLSR.Identity, rFOrLSR.NonLinearity( "abs", f = tor.abs ) ] # List of NonLinearity objects, must start with identity

RegMat, RegNames = rFOrLSR.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = rFOrLSR.CTors.NonLinearizer( None, RegMat, RegNames, NonLinearities ) # add the listed terms to the Regression matrix
```

<br/>

# 3. Validation
This example uses the `rFOrLSR.DefaultValidation` ([see Tutorial 1](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/1_Linear_in_the_Parameters)). The only difference between each channel's validation (the example assuming that both channels contain similar terms and thus share the same dictionary $D_C$) is the $y$ variable.  
Note that deep-copies are necessary to avoid python overwriting the same dictionary.

<br/>

# 4. Fitting
One arborescence is generated for each output $y_i$ and the correct dictionaries and $y$ vectors are passed.  

**Note 1 (RAM):** For large arborescences it is recommended to deletes old arborescences before fitting a new one.

**Note 2 (Printing):** `PlotAndPrint` is called on both arborescences after both have been fitted, such that the results are directly beneath each other in the console:  

```
7 Terms yielding an Mean absolute Error (MAE) of 3.973e-15% and a maximal deviation of 4.000e-14% and a Median Absolute Deviation (MAD) of 3.3332e-15

Recognized regressors:
-0.2999999999999999 x2[k-2]^2 * y2[k-1]
-0.8000000000000002 abs(x3[k-1] * y1[k-2])
-0.6999999999999997 x1[k-1] * x2[k-1]^2
 0.4999999999999999 x1[k-2] * x2[k-3]
 0.29999999999999993 x2[k]^3
 0.7 abs(x3[k])
 0.20000000000000007 x1[k]


7 Terms yielding an Mean absolute Error (MAE) of 2.547e-15% and a maximal deviation of 3.521e-14% and a Median Absolute Deviation (MAD) of 2.2005e-15

Recognized regressors:
-0.9 abs(x3[k-1] * y2[k-2])
-0.40000000000000013 x3[k-2]^2 * y1[k-1]
-0.6999999999999998 x2[k-1]^2 * x3[k-1]
 0.6000000000000001 x1[k-2] * y1[k-3]
 0.5 x3[k]^3
 0.7000000000000001 abs(y1[k-1])
 0.30000000000000004 x1[k-1]
```

[Previous Tutorial: 5. Sigmoid Expansion & Custom Validation](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/5_tanh)  
[Next Tutorial: 7. MaxLags](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/7_MaxLags)
