# Example 3: Imposing Regressors
This example / tutorial illustrates:
- How to impose regressors via $D_S$ (*D*ictionary of *S*elected regressors)
- The (linear) IIR (ARX) analysis tools: how to transform the regression results ($\theta, L$) into the $\underline{b}$, $\underline{a}$ coefficient-vectors and plot the resulting IIRs (magnitude, phase, poles/zeros)

<br/>
<div align="justify">

# 1. Test Setting

This example generates 5 random biquadratic peak filters and applies them in parallel to the input sequence to emulate a resonant linear system with 5 resonances. The resulting system is then fitted with the rFOrLSR as a single IIR filter, thus only linear regressors are used: $x\[k-j\], j \in \[k\]\_{k = 0}^{q\_x}$ and $y\[k-j\], j \in \[k\]\_{k = 1}^{q\_y}$ where the maximum delays / lags $q_x,\ q_y$ are set to be twice the chosen number of biquads.


``` python
def RandomPeakIIR( FreqRange, QRange, Fs ):
  f0 = RNG.integers( int( FreqRange[0] ), int( FreqRange[1] ) )
  Q = RNG.uniform( QRange[0], QRange[1] )

  b, a = sps.iirpeak( f0, Q, Fs )

  return ( b / a[0], a / a[0] ) # return normalized coeffs

Filters = [ RandomPeakIIR( [10, 8_000], [15, 30], Fs ) for _ in range( nBiquads ) ]

def Sys( x, Filters, Print = False ):
  """Applies the passed filters in parallel to the input signal"""
  if ( len( Filters ) < 1 ): raise ValueError( "There must be at least one filter" )

  if ( Print ): 
    for i in range( len( Filters ) ):
      print( "Biquad", i + 1, ":\nb =", Filters[i][0], "\na =", Filters[i][1] )
    print( "\n" )
  
  y = np.zeros_like( x.cpu().numpy() )

  for i in range( len( Filters ) ): # apply in parallel
    y += sps.lfilter( Filters[i][0], Filters[i][1], x.cpu().numpy() )

  return ( tor.tensor( y ) ) # recast to PyTorch tensor, which also


# Generate x and y data
x = tor.tensor( RNG.uniform( - Amplitude, Amplitude, size = p ) ) # uniformly distributed white noise
x -= tor.mean( x ) # CENTER !!!
y = Sys( x, Filters ) # Apply selected system
```

<br/>

# 2. Training Data

Since this example fits a (linear), only lagged regressors are needed: $x\[k-j\], j \in \[k\]\_{k = 0}^{q\_x}$ and $y\[k-j\], j \in \[k\]\_{k = 1}^{q\_y}$.

``` python	
y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) ) # Create the delayed regressors (cut to q to only have swung-in system)
```

<br/>

# 3. Validation Data

The validation dataset is created as in previous examples / tutorials for the `rFOrLSR.DefaultValidation`. The only difference here is that the "Data" entry is set to None, since no $D_C$ is passed to chose regressors from, as we're only imposing regressors. The imposed dictionary is passed to the validation function via the `DsData` dictionary entry. Since we're fitting an (linear) IIR, `ExpansionOrder` is set to 1, `NonLinearities` is set to only contain the identity and `MakeRational` is set to None.

``` python	
DsValDict = { # contains essentially everything passed to the CTors to reconstruct the regressors
  "y": [],
  "Data": None, # No free to chose from regrssors in this example
  "InputVarNames": [ "x", "y" ], # variables in Data, Lags, etc
  "DsData": [],
  "Lags": ( qx, qy ),
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": [ rFOrLSR.Identity ], # no non-linearities
  "MakeRational": None, # equivalent to [ False ]
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  x_val = tor.rand( int( p / 3 ) ) # Store since used twice
  x_val -= tor.mean( x_val ) # CENTER!!!
  y_val = Sys( x_val, Filters ) # _val to avoid overwriting the training y

  y_val, DsData, _ = rFOrLSR.CTors.Lagger( Data = ( x_val, y_val ), Lags = ( qx, qy ) ) # RegName not needed since same as training
  DsValDict["y"].append( y_val ) # cut version by Lagger
  DsValDict["DsData"].append( DsData )
```
<br/>

# 4. Fitting

Since we're only imposing regressors: no `Dc`, no `DcNames`, no tolerances, no `MaxDepth` need to be passed as arguments.

``` python	
Arbo = rFOrLSR.Arborescence( y,
                             Ds = RegMat, DsNames = RegNames, # Ds & Regressor names, being dictionary of selected regressors
                             ValFunc = rFOrLSR.DefaultValidation, ValData = DsValDict, # Validation function and dictionary
                           )
```
**Note (Root only):** Since this example only consists of imposed regressors, no arborescence is spanned, the root is constructed and the validation is performed. The arborescence is designed improve the term selection, but “no touchy” to whatever the user imposes. If a mixture of imposed and candidate regressors is passed ($D_S$ and $D_C$), then the arborescence will only optimize its $D_C$ selection.

**Note (Tolerance):** When imposing regressors, the tolerances are not taken into account, in the sense that the regression not will stop mid- $D_S$ fitting if the threshold is reached. Imposed means imposed. All regressors imposed by the user will be in the regression. This also holds when using $D_S$ in conjunction with $D_C$ (not shown here), as the entire $D_S$ will be added to the regression, after which (if the error is above the threshold) regressors from $D_C$ are added to the regression. Here since only imposed regressors are used in the regression, the threshold is not taken into account and thus none are passed.

<br/>

# 5. IIR Tools

This section describes the IIR tools provided by the submodule `rFOrLSR.Tools` used in this example. Further tools will be discussed in the next tutorials.

## 5.a. Reconstructing the IIR coefficient vectors $\underline{b}$, $\underline{a}$

First, we transform the regression results into the $\underline{b}$, $\underline{a}$ coefficients via the `rFOrLSR.Tools.rFOrLSR2IIR` function:

Arguments are the regression coefficients `theta`, the selected regressors indices `L` and regressor names `RegNames`.
In this example, since no regressors are chosen from $D_C$, the returned `L` is empty and must thus be reconstructed to index `RegNames`. Here, since all generated regressors are used, a for-loop (or np.arange) of the length of `theta` (or equally valid `RegNames`) suffices.

**Attention:** Not overwriting RegMat and RegNames with the arborescence's output is generally a bad move since a) the Arbo-CTor does duplicate removal, which is necessary for the rFOrLSR to work properly and b) the dictionary might have been morphed (impossible here since $D_S$ is never morphed).
In particular, this is important if either `RegNames` or the dictionary $D_C$ content is re-used by the user with `L`, which is the case in this example. However, for linear IIRs, no morphing is applied (no non-linearities are used) and lagging regressors shouldn’t generate duplicate regressors.


``` python
b_Ds, a_Ds = rFOrLSR.Tools.FOrLSR2IIR( theta, [i for i in range( len( theta ) )], RegNames ) # reconstruct L since empty as no terms selected, here all are taken in order`
```

## 5.b. IIR Plot
The IIR_Spectrum plot shows the magnitude and phase response of the list (iterable) of passed filters.  

`rFOrLSR.Tools.IIR_Spectrum` also takes iterables of $\underline{b}$, $\underline{a}$ coefficient-vectors but here the frequency response of the original filter must be computed, so we're transforming the $\underline{b}$, $\underline{a}$ coefficient vectors into the frequency response via the `scipy.signal.freqz` function.

Also the `xLims` are limited to be between 10Hz and half of the sampling rate as is commonly done in audio signals.


``` python
Resolution = 5_000
w, h_Ds = sps.freqz( b_Ds, a_Ds, worN = Resolution, fs = Fs )

h_Original = np.sum( [ sps.freqz( filt[0], filt[1], worN = Resolution, fs = Fs )[1] for filt in Filters ], axis = 0 ) # sum since parallel filters

# Plot frequency responses
rFOrLSR.Tools.IIR_Spectrum( h_List = [ h_Original, h_Ds ], FilterNames = [ 'Original', 'Estimated' ], # what to plot
                               Fs = Fs, # Frequency of sampling (aka Sampling rate)
                               xLims = [ 10, Fs / 2 ], # x-limits common to both plots
                               yLimMag = [ -50, None ] # stop at -50dB but make upper-limit data dependent with None
                             )                          
```
The below figure shows the resulting magnitudes and phases and illustrates that the estimated filter perfectly overlaps with the original filter, denoting an excellent fit

![Figure3](https://github.com/Stee-T/rFOrLSR/blob/main/Examples/3_Only_Ds_and_IIR/Figure_3.png)

**Note (Phase offsets):** In some cases, the fitted filter has a phase offset of exactly $\pm \pi$ radians (despite a correct magnitude response). It's a bit unclear to me why this happens but this can trivially be compensated by multiplying the $\underline{b}$ and $\underline{a}$ vectors by -1 (except of course $a_0$). If anyone has an idea to why that happens, please let me know, IIRs are not my specialty :)

## 5.c. Z-Plane Plot
The Z-plane plot shows the poles and zeros around the unit circle of the passed filters in $\underline{b}$, $\underline{a}$ format.  
The red crosses are the poles and the green circles are the zeros.

``` python
rFOrLSR.Tools.zPlanePlot( b_Ds, a_Ds, Title = 'Estimated' )
```

![Figure4](https://github.com/Stee-T/rFOrLSR/blob/main/Examples/3_Only_Ds_and_IIR/Figure_4.png)

<br/>

[Previous Tutorial: 2. Rational Fitting](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/2_Rational_Fitting)  
[Next Tutorial: 4. Sigmoid Expansion & Custom Validation](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/4_tanh)
