# Generalities
<div align="justify">

This GPU-accelerated Python package contains the machine learning algorithms described in my two papers "*Arborescent Orthogonal Least Squares Regression (AOrLSR)*" and "*Dictionary Morphing Orthogonal Least Squares Regression (DMOrLSR)*" (coming soon) both based on my "*Recursive Forward Orthogonal Least Squares Regression (rFOrLSR)*" to fit "*Non-Linear Auto-Regressive Moving-Average Exogenous input systems (NARMAX)*". So, now that we have covered all the fancy acronyms, we might get into some explanations. Otherwise jump straight into the [library examples](Link "Example folder").

**Note 1:** The library currently only implements the arborescence part (see below) and is thus not finished, missing the dictionary morphing part. This means that currently only static regressors can be fitted. Thus, the dictionary terms need to be pre-defined and are not adapted to the system by the regression.

**Note 2:** Github's Latex engine is unreliable, so please forgive that certain expressions (especially sums and underlines) are not rendered properly or not at all. All $x$ and $\chi$ in in matrix equations are of course vectors and should be underlined (check the readme.md if in doubt).

<br/>

# NARMAX...who?

**• N for Non-linear:** Any non-linearity (functions or piecewise definitions) or non-linear combination method (products of terms, etc.) applied to any of the following types of terms.

**• AR for Auto-Regressive:** Any type of feeding the system-output $y$ back into the system, via feedback or recursion. Thus, anything containing  temporal terms (with $y\[k-j\]$ terms such as $\tanh(0.5y\[k-1\] + 0.3y\[k-2\]y\[k-3\])$ or $y^a\[k-1\]x^b\[k-3\]$) or spatial $y$ terms ($y_1y_2y_3$) or any spatio-temporal combination.

**• MA for Moving Average:** Any type and distribution of internal noise terms $e\[k\]$, which accounts for fitting error or any variables not present in the system input. Can also represent internal random states spanning multiple time steps or spatial distributions.

**• X for exogenous input:** Any type of system-input terms $x$ which can be temporal ($x^n\[k-j\]$ or $e^{x\[k\]y\[k-2\]}$) or spatial ($x_1x_2x_3$, etc) or spatio-temporal (a mixture of both).

A NARMAX system contains thus any arbitrary mixture of the above terms. The provided library examples (being those in the AOrLSR paper) demonstrate the fitting of the below dynamic (=temporal) systems. Those have no other special properties than being stable for an input of amplitude 1 (or potentially more) and being the most ridiculously non-linear systems I came up with at that time to demonstrate the fitting power and flexibility of the algorithms.


### 1. Linear-in-the-parameter Example
$y\[k\] = 0.2x\[k\] + 0.3x^3\[k-1\] + 0.7|x\[k-2\]x^2\[k-1\]| +0.5e^{x\[k-3\]x\[k-2\]} - \cos(y\[k-1\]x\[k-2\]) -0.4|x\[k-1\]y^2\[k-2\]| - y^3\[k-3\]$

This is essentially a monomial expansion of IIR terms ($y[k-j]$ and $x[k-j]$) also passed through common non-linearities such as abs, cos and exp, yielding a heavily non-linear NARX system.  
[Code for this example](Link "Standard Example")

### 2. Rational Example
$y\[k\]=\frac{0.6|x\[k\]|-0.35x^3\[k\]-0.3x\[k-1\]y\[k-2\]+0.1|y\[k-1\]|}{1-0.4|x\[k\]|+0.3|x\[k-1\]x\[k\]|-0.2x^3\[k-1\]+ 0.3y\[k-1\]x\[k-2\]}$

This demonstrates that (for NARX systems) rational non-linear models can be fitted by linearizing the terms: $y\[k\]=\frac{A}{1+B}\iff y\[k\](1+B)=A⟺y\[k\]=A-y\[k\]B$, $A$ and $B$ being linear-in-the-parameter systems such as system 1 in the above example.  
[Code for this example](Link "Rational Example")

### 3. Expansion-in-an-expression Example
$y = \text{sgn}(x)(1-\frac{1}{1+|x|A})$ with $A≔\Sigma_{j\in J}\theta _j |x|^j$ and $J\subseteq \mathbb{N}$


This is a memoryless NX (Non-linear exogenous input) system aka a normal scalar function, depending only on $x$. This system shows that NARMAX expansions can be inserted into expressions to impose constraints or system properties (here quick convergence to $\text{sgn(x)}$ and low error around the origin) or obtain complex fitting. This specific expansion is designed to emulate tanh with another continuous rational function. The provided code also demonstrate how to chose the number of terms in such an expansion.   
[Code for this example](Link "tanh Example")

<br/>


**The NARMAX fitting steps:**  

1) **Expansion Type Selection:** As usual in machine learning, one must first chose an appropriate expansion type (FIR, IIR, Monomially-Expanded IIR, RBF, Wavelet, arbitrary non-linearities, etc.). As expected, the model quality strongly depends on how relevant the chosen expansion is. The advantage of this library's rFOrLSR is that any type of expansion and any mix of expansions is supported, as the rFOrLSR is based on vector matching methods.  
This is achieved by creating a fitting "dictionary" $D_C \in \mathbb{R}^{p \times n_C}$ (Pytorch matrix) containing the candidate regressors $\underline{\varphi}_k \in \mathbb{R}^{p}$ stacked column-wise and passing it to the library. 

2) **Model Structure Detection:** The first real challenge is to find the correct regressors from all those present in the user-defined dictionary $D_C$, as most system behaviors can be sufficiently well described with very few terms.  
To illustrate, the first system above contains a single cosine term which needs to be retrieved from the set of cosines with relevant monomial expansions as arguments.  

3) **Model Parameter Estimation:** Finally, once the correct expression terms are selected, their regression (= scaling) coefficients must be chosen. The rFOrLSR's optimization criterion is least squares.

<br/>

# About the library

As described above, the library makes optimal sparse least squares fitting of any vectors / regressors $\underline{\varphi}_k \in \mathbb{R}^{p}$ passed by the user for the given system response $\underline{y}\in \mathbb{R}^{p}$. All provided examples are with severely non-linear auto-regressive systems, however, the user can pass dictionaries containing any type of functions (multivariate, discontinuous, discrete, acausal, etc. - whatever that outputs vectors with numbers really). This is thus a very general mathematical framework supporting for example FIR/MA, IIR/AR, wavelet-, RBF, polynomial, etc. fitting.

## Installation & Usage
**Installation:** The library is currently unfinished and has thus not yet been made available through pip install. This will be done soon when the morphing part is finished. Same for the docs, in the mean time, please refer to the examples (see NARMAX section above), which are representative of the average usage and well commented.

**Usage:** Currently, the library folder can be downloaded and copy-pasted in the same folder as the script calling it. The import is as usual for python packages.


## Library content

### Signal Constructors
The signal CTors allow to easily generate regressors to be h-stacked in the dictionary $D_C$ passed to the arborescence. They cover the most common types of expansions like FIR/MA, IIR/AR, Power-series, oscillations, RBF, Wavelet, etc. Any mixture and any custom user function/regressor can be put in the dictionary to be used for fitting.

* **Lagger:** Creates delayed $\underline{x}, \underline{y}$ and optionally $\underline{e}$ terms up to the desired order.  
This is the only CTor needed for linear FIR and IIR systems.  
Example: $\underline{x} → \[\ \[\underline{x}\[k-j\]\ \]_{j=0}^{n} = \[\ \underline{x}\[k\], \underline{x}\[k-1\], \underline{x}\[k-2\],...\ \]$

* **Expander:** Monomially expands the input by creating all product combinations up to a certain combined power of the passed input.  
Example: Set of two terms with 3rd order expansion: $\[\underline{x}_1,\ \underline{x}_2 \]→\[\ \underline{x}_1,\ \underline{x}_2,\ \underline{x}_1^2,\ \underline{x}_2^2,\ \underline{x}_1\cdot\underline{x}_2,\ \underline{x}_1^3,\ \underline{x}_2^3,\ \underline{x}_1^2\cdot\underline{x}_2,\ \underline{x}_1\cdot\underline{x}_2^2\ \]$

* **NonLinearizer:** Applies non-linearities to the input in addition to allowing terms to be in the denominator for rational functions.  
Note that this is limited to elementwise functions, other fancy things must currently be done by hand.  
Example: $\underline{x}→\[\ \underline{x}, f_1(\underline{x}), f_2(\underline{x}), \frac{1}{f_3(\underline{x})},...\ \]$

* **Oscillator:** (Coming soon)  
Constructor for common oscillations like cos, sin with options for different frequencies, phases and amplitude decays.

* **RBF:** (Coming soon)  
Constructor for common Radial Basis Functions

* **Wavelet:** (Coming soon)  
Constructor for common wavelet functions

* **Input signal Generators** (Coming soon)  
Probably something facilitating the construction of input signals $\underline{x}$ like Maximum Length Sequences (MLS), multi-level sequences, etc.

* Whatever anyone feels like contributing like Chirps or orthogonal bases for specific types of analysis :)


### Analysis tools

* **MaxLagPlotter:** Tool designed to estimate the maximum lags and the expansion order CTor for $\underline{x}$ and $\underline{y}$ terms via polynomial NARMAXes.  
Uses:
  * Allows automating the selection of the parameters passed to the *Lagger* and *Expander* CTors for polynomail fitting.  

  * Helps eliminating unnecessary regressors from the dictionary before starting the arborescence for great speed-ups.   
  (Note that this is still under development and in some cases much slower than running the arborescence with a large dictionary.)

  * Allows to analyze the percentage of explained variance per lag and per expansion order. 

* **NonLinearSpectrum:** (Coming soon).  
Essentially a Fourier transform for Nonlinear systems based on polynomial NARMAXes.

### rFOrLSR & Arborescence
The library's main object is the arborescence which takes in the user-defined dictionaries (imposed regressors $D_S$ and candidate regressors $D_C$) and the desired system input $\underline{y}$ for the fitting. The arborescence, being a breadth-first tree search in the solution space spanned by the dictionary $D_C$ columns, repeatedly calls the rFOrLSR to iteratively refine the solution. The more "levels" the arborescence has, the more search space is traversed and the higher the probability of finding the optimal solution.  
For more on that, see the below "*More on the Algorithms*" section.


### Dictionary Morphing
**→ COMMING SOON**  
Essentially, an extension of the framework where the passed regressors are adapted to the system in real time by the rFOrLSR under very mild conditions. To illustrate, the user can pass any elementwise function $f\_⦿$ and the arborescence will morph it by adding scaling coefficients $\xi_j$ and arbitrary linear combinations of dictionary terms as arguments. The morphing is thus of the form:
$f\_⦿ \left(\underline{\chi}\_0 \right) \rightarrow f\_⦿ \left(\Sigma\_{j=0}^r \xi_j \underline{\chi}_j\right)$ 

Examples: 
* **Frequency and phase estimation:**  
$\cos(x) → \cos(2.1x + 0.2)$ with $\xi_0 = 2.1$, $\xi_1 = 0.2$,  $\underline{\chi}_1 = \underline{1}$

* **Wavelet translation and dilation:**  
$\psi(x) → \psi(ax + b)$ with dilation coefficient $\xi_0 = a$ and translation coefficient $\xi_1 = b$,  $\underline{\chi}_1 = \underline{1}$

* **Radial Basis Function arguments:**  
$\varphi(\left\|x_0 - c_0\right\|) → \varphi\left(\lVert \Sigma\_{j=0}^r (a_j x_j-b_jc_j) \rVert \right)$ with scaling factors $\xi_j = a_j$ and $\xi_j = b_j$,  $\underline{c}_j = \underline{1}$

* **Arbitrary elementwise functions**  
Arbitrary functions where changing coefficients or adding arguments is of interest.
Examples include oscillations where the decay / frequency needs to be varied, chirps whose speed / start or end frequency must be varied etc.

<br/>

# More about the algorithms

## Recursive Forward Orthogonal Least Squares Regression - rFOrLSR
The rFOrLSR (described in the first section of the AOrLSR paper) is a recursive matrix form of the classical FOrLSR (Forward Orthogonal Least Squares Regression) algorithm. Thus, the double for-loop inside the FOrLSR is replaced by a single matrix operation, which is recursively updated. The matrix-form allows the large-scale BLAS-like or GPU-based optimizations offered by this library, while the recursivity flattens the FOrLSR’s complexity from quadratic in model length $n_r$ ( $O(n_r^2)$ ) to linear ( $O(n_r)$ ). Then, the remaining FOrLSR procedure is rearranged and vectorized, in addition to allowing to efficiently impose regressors (which is not a native FOrLSR feature). These optimizations greatly reduce its computational cost, allowing use it as node processing procedure in large search trees like the proposed arborescence.

The (r)FOrLSR is a greedy algorithm selecting iteratively the regressors the most similar to the desired system output $\underline{y}$ to form the NARMAX model. After each regressor is added, to avoid describing the same system output information (aka $\underline{y}$ variance) repeatedly, the information is removed (via orthogonalization) from the remaining regressors in the dictionary of candidates $D_C$.  
The (r)FOrLSR is thus a greedy algorithm, since at each iteration, it takes the locally best choice, maximizing the increase in explained output variance without considering the global search space, which could contain shorter and/or lower validation error models. When a suboptimal choice is made, the following regressors are often chosen to compensate for that error, leading to poor models. For this reason, an arborescence spanning a large search space is proposed by my papers/library, see below.

## Arborescent Orthogonal Least Squares Regression - AOrLSR
The AOrLSR (described in the second section of the AOrLSR paper) mitigates the (r)FOrLSR's greediness by tactically imposing regressors in the model and aborting or skipping regressions once their results become predictable. My paper proposes several theorems and corollaries dictating which regressions must be computed and which not. Most of the time, only a marginal fraction (<0.5-5%) of the search space must be computed to be traversed. This allows to find expansions with fewer regressors (sparse fitting) and/or lower error by choosing different regressors.

The arborescence also provides a validation mechanism to select the best model. Once the search space is traversed (or if the search is ended manually), the validation procedure iterates over all computed models and runs the user-defined validation function on the models with the smallest number of regressors (sparsity criterion), diminishing overfitting and model computational expense. The arborescence object accepts a function pointer and a dictionary for the validation, allowing arbitrary validation methods on arbitrary validation data, such that any user-defined criterion can be used, such as signal mean-square error, frequency-domain similarities, computational expense estimations, etc.

## The AOrLSR as sparsifying solver
The (r)FOrLSR and the AOrLSR are, on an abstract level, linear equation system solvers, maximizing the solution-vector sparsity by choosing the most relevant LHS columns, while minimizing the mean squared error with respect to the RHS $\underline{y}$.
The LHS is $M≔D_S⫲D_C$, where $D_S$ allows to impose columns to the selection while the AOrLSR is free to choose any desired columns from $D_C$. $M$’s dimensions being arbitrary, the system can be under-, well- or over-determined.  

To illustrate, be the following system with $p \in \mathbb{N}^*$: $M \underline{x} = \underline{y}$

The AOrLSR with a $1 \%$ error tolerance could chose the regressor (here columns of $M$) index-set $\mathcal{L}=\{1,3\}$ and $\underline{\hat{\theta}} \in \mathbb{R}^{|\mathcal{L}|=2}$, such that $\underline{x}\[\mathcal{L}]:=\hat{\underline{\theta}}\iff \underline{x}=\[\hat{\theta}_1, 0, \hat{\theta}_2, 0,0\]^T$, while a $0.1 \%$ tolerance could yield $\mathcal{L}=\{1,3,4\}$ and $\underline{\hat{\theta}} \in \mathbb{R}^{|\mathcal{L}|=3}$, thus $\underline{x}=\left\[\hat{\theta}_1, 0, \hat{\theta}_2, \hat{\theta}_3, 0\right\]^T$.

The linear equation system is thus reduced to the most relevant  subset of $M$'s columns and $\underline{x}$ entries to describe the RHS $\underline{y}$: Thus, $\[ \underline{m} \_j\]\_{j \in \mathcal{L}} \underline{\hat{\theta}}=\underline{y}$ is solved via a QR-like decomposition.


## Dictionary Morphing Orthogonal Least Squares Regression - DMOrLSR

**COMMING SOON**

This part of the library is described in my "*Dictionary Morphing Orthogonal Least Squares Regression (DMOrLSR)*" paper. Many regressors of interest are (elementwise) non-linearities taking one or more arguments $f_⦿(\xi)$. It is, however, very unlikely that the passed candidate dictionary $D_C$ contains the optimal non-linearities of the form $f_⦿\left(\Sigma\_{j=1}^r \xi_j \underline{\chi}\_j\right)$, as this would require an quasi-infinite number of infinitesimal increments for each $\xi_j$ and a combinatorial order of $\underline{\chi}\_j$ combinations for each $r>0$. This would overflow RAM (see example 1 in the paper) and take insane computing times due to the enormous candidate dictionary $D_C$. It is thus more efficient to scan $D_C$ for the most fitting regressor and add arguments and coefficients to adapt the regressor to the system.  
(See examples in the morphing section above) 

For comparison, classic expansions like Fourier and Laplace fit the entire given dictionary (= imposed terms and expansion order), whereas the (r)FOrLSR selects only the most relevant terms (= system dependent term selection and expansion order). The next logical step is adapting the selected regressors to the system, making the expansion fully system dependent.

The morphing procedure comprises four steps. After the rFOrLSR regressor selection, the regressor is parsed to deduce the non-linearity $f$ and its argument ${\underline{\chi}}_0$. If the function is morphable, the procedure continues with a genetic vector-space generating algorithm (GenVSGen) determining how many and which supplementary arguments ${\underline{\chi}}\_j$ improve the fitting. Finally, an infinitesimal optimizer determines the optimal coefficients $\underline{\xi}$.
Once the rFOrLSR has selected and morphed sufficiently many regressors to meet its precision threshold, the final infinitesimal optimizer adapts the regression coefficients $\underline{\hat{\theta}}$ and the respective function arguments $\underline{\xi}_j$.  

The morphing works with closed-form solutions of the gradient and Hessian, such that no AutoGrad is needed, allowing many orders of magnitude faster fitting procedures on arbitrary (discontinuous, acausal, etc.) functions. 

The morphing allows:  
- Fitting much more complex expressions (= more expressive models)
- Sparser fitting (= smaller resulting expressions for a desired error tolerance)
- Lower error (= smaller mean squared error for same length expressions)
