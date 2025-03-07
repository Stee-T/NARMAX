<div align="justify">

# Example 5: Sigmoid Expansion & Custom Validation Function

**This example / tutorial illustrates:**
- How to fit nested expressions by linearizing the regressors
- How the validation function reflects the desired task ($L_{\infty}$ norm = max deviation rather than $L_2$ norm = least squares)
- The creation of a custom validation function

## 1. Training Data Creation
To illustrate the framework's flexibility, this tutorial demonstrates how to approximate sigmoids via a custom expansion (NARMAX nested inside a complex expression) of the following form: $y=\mathrm{sgn}\left(x\right)\left(1-\frac{1}{1+\left|x\right|A}\right)$ with $A$ being a (ideally sparse) weighted sum of powers of $|x|$, yielding $A≔\Sigma_{j\in J}\left|x\right|^{j}$ for $J\subset\mathbb{N}_0$.

First, one must determine the equation's RHS and LHS, since the rFOrLSR fits expressions of the form $\underline{y}= D_C \cdot \underline{\theta}$.   
The above expansion must thus be linearized:  

$y=\mathrm{sgn}\left(x\right)\left(1-\frac{1}{1+\left|x\right|A}\right)
\iff \frac{y}{\mathrm{sgn}(x)} -1 = \frac{-1}{1+\left|x\right|A} \iff -\mathrm{sgn}\left(x\right)y+1=\left(\mathrm{sgn}\left(x\right)y-1\right)\left|x\right|A $

**Note (Expansion Start):** For convenience, in the RHS, $\left|x\right|$ is merged with $A$, such that the expansion starts at $\left|x\right|$ (thus $j=1$) rather than at $1$ ($j=0$).

Thus, the LHS of the regression equation is $-\mathrm{sgn}\left(x\right)y+1$ and the RHS is $(\mathrm{sgn}\left(x\right)y-1)\left|x\right|A$ yielding
$D_C := \[ \left(\mathrm{sgn}\left(x\right)y-1\right) \left|x\right|^j \]_{j\in \mathbb{N}^*}$.  
The powers of $|x|$ and the corresponding name-strings are conveniently generated by `NARMAX.CTors.Expander`. The data is expected to be a 2D-torch.tensor, so we transpose $\underline{x}$ before passing it to the function. Finally, the remaining operations computed above are performed on the data.

``` python
RegMat, RegNames = NARMAX.CTors.Expander( Data = tor.abs( x.view( -1, 1 ) ), RegNames = [ "|x|" ], ExpansionOrder = ExpansionOrder )
y = - y / tor.sign( x ) + 1 # subtract sign(x) to impose it in regression and divide by it
RegMat = -y.view( -1, 1 ) * RegMat # multiply by -y to put regressors in denominator
```

## 2. Validation Handling
Validation is crucial to assess the performance and generalizability of a fitted model. Models performing well on the training sequence might be overfitted and not represent the underlying system well and thus yield poor performance on unseen inputs. The arborescence generates many models and its validation procedure only selects the ones with the minimum number of regressors as candidate models.

This section demonstrates how to create a custom validation function and its associated validation dictionary.  
There is not much to say about the validation dictionary other than it must contain whatever the validation function requires.

When the training data is custom-made (in the sense of not constructible via the 3 basic regressor CTors `Lagger`, `Expander` and `NonLinearizer`), a custom validation function must be provided (see Note below). Custom validation functions must perform three operations:

**A) Regressors Creation:** TL;DR: Save GPU-RAM by creating the validation dictionary on the fly to allow for larger $D_C$ and more validation sets.  
The best practice is having several validation data-sets (input noise sequences $\underline{x}$), however, the GPU-RAM would need to hold the training $D_C$ and all the validation $D_C$ simultaneously, which limits the dictionary size. It is thus recommended that the validation function creates the validation regressor dictionary on the fly, to save GPU-RAM to allow a larger $D_C$ (see Note below).   
The arborescence only validates regressions of minimum length (number of regressors), of which there are mostly few, in addition to filtering out regressions made of permutations of regressors. Thus, (for most cases) the computation cost of creating the dictionary for each to-be-validated regression is negligible compared to the gain in fitting potential won by using larger $D_C$.

However, the only constraint on the validation function is taking the imposed 6 input arguments (See below). The validation function being arbitrary, it could just use $\underline{\hat{\theta}}$ and $\underline{L}$ to index pre-computed dictionaries, if either enough RAM is available or if the dictionary construction is computationally expensive.

**B) Compute Output Estimate $\underline{\hat{y}}$ :** For linear-in-the-parameter systems, this is trivially the selected columns of the regression matrix multiplied with the estimates regression coefficients $\underline{\hat{y}} = \left\[\underline{d}\_j\right\]\_{j\in L} \cdot \underline{\hat{\theta}}$, which is code yields:

``` python
yHat = RegMat[ :, L.astype( np.int64 ) ] @ tor.tensor( theta )
```

The arborescence compresses $\underline{L}$ to be in the smallest possible integer type (to save CPU-RAM, since millions of regressions are stored) but only 64-bit floats are currently valid python indices. Theta is a numpy array, so it needs casting for the mat-mult.

For not linear-in-the-parameter systems, the validation function needs to insert the above linear combination in the remaining equation, see below example.

**C) Compute desired error $E\left(\underline{\hat{y}},\underline{y}\right)$:** This is where the actual validation procedure happens. To quote my paper "*The validation score is application dependent and should test important model characteristics. To illustrate, the validation could, additionally to the ERR, operate in the frequency domain, and also penalize models based on their computational expense. The validation can also reject models based on arbitrary criteria like using negative or too large regression coefficients.*"  
In this example, the validation function could also for example penalize models based on the symmetry of the generated function.

**Note (Memory-Management):**  
**Q:** *Why not just pass pre-made data to the validation function instead of providing a function generating it?*  
**A:** This library is designed to be maximally scalable, so (on a consumer PC's GPU) $D_C$ might already take a large part of the GPU-memory. The library encourages users to use many different validation data-sets, which increases GPU-RAM scarcity. Furthermore, to minimize GPU-RAM usage, the index-sets computed by the arborescence are stored in (CPU-)RAM.

### 2.A. Validation Dictionary
This example, being an approximation for a scalar function, requires the validation to select the model with the smallest maximal deviation ($L_{\infty}$ norm) rather than using the usual least squares criterion ($L_2$ norm). Thus, the validation sequence is a high-resolution linspace, such that the function's input domain is equidistantly sampled.

The validation dictionary must only contain the data needed by the validation function:

``` python
xValidation = tor.linspace( -TestRange, TestRange, 50_000 )
yValidation = tor.tanh( xValidation )

ValidationDict = { # Contains only the data used by our custom validation function
  "y": [ yValidation ], # desired output
  "Data": [ xValidation.view( -1, 1 ) ], # linspaces to compute the function
  "ExpansionOrder": ExpansionOrder, # powers of |x|
}
```

### 2.B. Validation Function
The validation function does some input checking after which the three aforementioned steps are executed.

**Input list:** Only ValDic needs to be taken care of, as all other arguments are generated by the arborescence.
* **theta:** (1D np.array) containing the estimated regression coefficients.
* **L:** (1D np.array) containing the indices of the selected regressors (not those in $D_S$ !).
* **ERR:** (1D np.array) containing the regression ERR values
* **ValDic:** (dict) containing the user-generated validation data
* **MorphDict:** (dict) containing the morphing meta-data, generated by the arborescence
* **DcFilterIdx:** (1D np.array) containing the unfiltered regressors' indices. The arborescence's constructor filters out duplicates (like $x^2$ vs $|x^2|$) from the dictionary, which must be mirrored by the validation to keep the indices in $L$ correct.

**Important aspects:**  
**A) For-loop:** Validation functions should ideally support multiple input sequences.

**B) Index-Filtering:** The Arborescence's constructor filters out duplicates from the dictionary, which must be mirrored by the validation to keep the indices in $L$ correct.  

**C) Centering:** The Arborescence's constructor centers the regressors in the dictionary before running the NARMAX. If this is to be mirrored by the validation is case dependent.


``` python
def Sigmoid_Expansion_L_inf( theta, L, ERR, RegNames, ValDic,  DcFilterIdx = None ): # The arborescence imposes 6 arguments to passed validation functions

  # --------------------------------------------------------------- Defensive programming ------------------------------------------------------------------
  if ( not isinstance( ValDic, dict ) ): raise AssertionError( "The passed ValDic datastructure is not a dictionary as expected" )
  
  for var in [ "y", "Data", "ExpansionOrder" ]: # no functions passed since done manually
    if ( var not in ValDic.keys() ): raise AssertionError( f"The validation datastructure contains no '{ var }' entry" )
  
  if ( not isinstance( ValDic["Data"], list ) ):  raise AssertionError( "ValDic's 'Data' entry is expected to be a list" )

  # ----------------------------------------------------------------------- Validation --------------------------------------------------------------------
  Error = 0 # total error
  
  for i in range( len( ValDic["Data"] ) ): # iterate over all passed Data tuples
    RegMat, _ = NARMAX.CTors.Expander( Data = ValDic["Data"][i], RegNames = [ "|x|" ], ExpansionOrder = ValDic["ExpansionOrder"] ) # create data

    if ( DcFilterIdx is not None ): RegMat = RegMat[:, DcFilterIdx] # Filter out same regressors as for the regression

    A = tor.abs( RegMat[ :, L.astype( np.int64 ) ] ) @ tor.tensor( theta ) # A abs-polynomial as in paper
    yHat = tor.sign( ValDic["Data"][i] ).view( -1 ) * ( 1 - 1 / (1 + A) ) # create Sigmoid with sign(x) * ( 1 - 1 / ( 1 + Expansion ) )

    Error += tor.max( tor.abs( ValDic["y"][i] - yHat ) ) # maximum absolute error
    
  return ( Error / len( ValDic["Data"] ) ) # norm by the number of validation-sets ( not really necessary for AOrLSR but printed for the user )
```

## 3. Fitting
Business as usual of initializing and calling fit():

``` python
Arbo = NARMAX.Arborescence( y,
                            Dc = RegMat, DcNames = RegNames, # dictionary of candidates and the column names
                            tolRoot = tol, tolRest = tol, # \rho tolerances
                            MaxDepth = 3, # Maximal arborescence depth
                            ValFunc = Sigmoid_Expansion_L_inf, ValData = ValidationDict, # Validation function and dictionary
                          )

theta, L, ERR, _, RegMat, RegNames = Arbo.fit()
```

## 4. Results inspection
Similarly to the validation function, a linspace is used to plot the error. The same construction method for $\underline{\hat{y}}$ is used as in the above validation function.

### 4.1 Printing the results

``` python
Expression = ""
for i in range( len( theta ) ): Expression += f"{ theta[i] } { RegNames[ L[i] ] } + "
print( "\n", Expression[ : -3 ], "\n" )
```

In general, any for-loop on the regressors should run over len( $\underline{\hat{\theta}}$ ) rather than len( $\underline{L}$ ) to take into consideration any potential user-imposed terms in $D_S$, as len( $\underline{\hat{\theta}}$ ) = len( $\underline{L}$ ) + nCols ( $D_S$ ).

### 4.2 Plotting

```python
x_Plot = tor.linspace( -TestRange, TestRange, 5_000 )

RegMat_Plot = NARMAX.CTors.Expander( Data = x_Plot.view( -1, 1 ), RegNames = [ "|x|" ], ExpansionOrder = ExpansionOrder )[0] # ignore the names

A = ( tor.abs( RegMat_Plot[ :, L ] ) @ tor.tensor( theta ) ).cpu()
x_Plot = x_Plot.cpu()

Fig, Ax = plt.subplots()
Ax.plot( x_Plot, tor.tanh( x_Plot ) - tor.sign( x_Plot ) * ( 1 - 1 / (1 + A) ) )
Ax.grid( which = 'both', alpha = 0.5 )
Ax.legend( ["Fitting error"] )
Fig.tight_layout()

plt.show()
```


<br/>

[Previous Tutorial: 4. Imposing Regressors & IIR fitting](https://github.com/Stee-T/NARMAX/tree/main/Examples/4_Only_Ds_and_IIR)  
[Next Tutorial: 6. Multiple Input Systems](https://github.com/Stee-T/NARMAX/tree/main/Examples/6_MIMO)