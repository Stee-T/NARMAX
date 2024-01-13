import torch as tor

import rFOrLSR

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs

import scipy.signal as sps
import numpy as np

# --------------------------------------------------------------------------------------- Hyper-Parameters ---------------------------------------------------------------------------------------
p = 2_000 # Dataset size
Amplitude = 1
ExpansionOrder = 1 # only allow terms to have exponent 1, thus linear
W = None # Gaussian white noise
Fs = 44_100 # Signal sampling frequency

nBiquads = 5
qx = 2 * nBiquads; qy = 2 * nBiquads # maximum x and y delays
print( "number of biquads:", nBiquads, "\nLags: qx:", qx, "qy:", qy )

# use a fixed seed to reproduce results
import numpy as np
Seed = 3 # fixed seed to reproduce results
print ( "\nSeed: ", Seed, "\n" )
RNG = np.random.default_rng( seed = Seed )

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
x -= tor.mean( x ) # Center!!!
y = Sys( x, Filters ) # Apply selected system

# ---------------------------------------------------------------------------------------  Training data ---------------------------------------------------------------------------------------
# We're fitting an IIR in this example, so we only need to create lagged regressors
y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) ) # Create the delayed regressors (cut to q to only have swung-in system)

print( "\nDict-shape:", RegMat.shape, "\nqx =", qx, "; qy =", qy, "\nExpansionOrder =", ExpansionOrder, "\n\n" )

# --------------------------------------------------------------------------------------- Validation data ---------------------------------------------------------------------------------------

DsValDict = { # contains essentially everything passed to the CTors to reconstruct the regressors
  "y": [],
  "Data": None, # No free to chose from regressors in this example
  "InputVarNames": [ "x", "y" ], # variables in Data, Lags, etc
  "DsData": [],
  "Lags": ( qx, qy ),
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": [ rFOrLSR.Identity ], # no non-linearities
  "MakeRational": None, # equivalent to [ False ]
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  x_val = tor.rand( int( p / 3 ) ) # Store since used twice
  x_val -= tor.mean( x_val ) # center
  y_val = Sys( x_val, Filters ) # _val to avoid overwriting the training y

  y_val, DsData, _ = rFOrLSR.CTors.Lagger( Data = ( x_val, y_val ), Lags = ( qx, qy ) ) # RegName not neede since same as training
  DsValDict["y"].append( y_val ) # cut version by Lagger
  DsValDict["DsData"].append( DsData )

# --------------------------------------------------------------------------------------- Fitting imposed regressors ---------------------------------------------------------------------------------------

# Since we're only imposing regressors: no Dc, no DcNames, no tolerances, no MaxDepth need to be passed
Arbo = rFOrLSR.Arborescence( y,
                             Ds = RegMat, DsNames = RegNames, # Ds & Regressor names, being dictionary of selected regressors
                             ValFunc = rFOrLSR.DefaultValidation, ValData = DsValDict, # Validation function and dictionary
                           )

theta, L, ERR, _, RegMat, RegNames = Arbo.fit()

Figs, Axs = Arbo.PlotAndPrint() # returns both figures and axes for further processing
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

b_Ds, a_Ds = rFOrLSR.Tools.rFOrLSR2IIR( theta, [i for i in range( len( theta ) )], RegNames ) # reconstruct L since empty as no terms selected, here all are taken in order

# --------------------------------------------------------------------------------------- Plots ---------------------------------------------------------------------------------------

Resolution = 5_000
w, h_Ds = sps.freqz( b_Ds, a_Ds, worN = Resolution, fs = Fs )

h_Original = np.sum( [ sps.freqz( filt[0], filt[1], worN = Resolution, fs = Fs )[1] for filt in Filters ], axis = 0 ) # sum since parallel filters

# Plot frequency responses
rFOrLSR.Tools.IIR_Spectrum( h_List = [ h_Original, h_Ds ], FilterNames = [ 'Original', 'Estimated' ], # what to plot
                               Fs = Fs, # Frequency of sampling (aka Sampling rate)
                               xLims = [ 10, Fs / 2 ], # x-limits common to both plots
                               yLimMag = [ -50, None ] # stop at -50dB but make upper-limit data dependent with None
                             )

rFOrLSR.Tools.zPlanePlot( b_Ds, a_Ds, Title = 'Estimated' )

plt.show()