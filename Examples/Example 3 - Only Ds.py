import torch as tor

import rFOrLSR_0_14 as rFOrLSR
import rFOrLSR_0_14.Test as Test_Systems

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs



# ---------------------------------------------------------------------------------------  Hyper Parameters ---------------------------------------------------------------------------------------
p = 2_500 # Dataset size
Amplitude = 1
tol = 0.0001 # Fitting tolerance in %, so multiply by 100
ExpansionOrder = 1 # only allow terms to have exponent 1, thus keep terms linear
W = None # Gaussian white noise
qx = 4; qy = 4 # maximum x and y delays
q = max( ( qx, qy ) ) # take the maximum lag to truncate the signals

Sys = Test_Systems.ARX # Example 1 in paper

# Generate x and y data
while ( 5 ): # 5 is the absolute truth, do while y contains no nan
  x = Amplitude * ( tor.rand( p ) * 2 - 1 ) # uniformly distributed white noise
  x -= tor.mean( x ) # center
  y = Sys( x, q, W, Print = True ) # apply selected system
  if ( not tor.isnan( tor.sum( y ) ) ): break

# ---------------------------------------------------------------------------------------  Training data ---------------------------------------------------------------------------------------
# No Non-linearities are needed, however the standard validation function needs this information, so the usual list must be created
NonLinearities = [ rFOrLSR.Identity ] # List of NonLinearity objects, must start with identity
MakeRational = [ False ] # List of bool, deciding if each NonLinearity is to also be made rational 

# ---------------------------------------------------------------------------------------  Training data ---------------------------------------------------------------------------------------
# We're fitting an IIR in this example, so we only need to create the lagged terms
y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x[q:], y[q:] ), MaxLags = ( qx, qy ) ) # Create the delayed signal terms (cut to q to only have swung-in system)

print( "\nTolerance: " + str(tol), "\nArboDepth:", 0, "\nDict-shape:", RegMat.shape, "\nqx =", qx, "; qy =", qy, "\nExpansionOrder =", ExpansionOrder, "\n\n" )

# --------------------------------------------------------------------------------------- Validation data ---------------------------------------------------------------------------------------

ValidationDict = { # contains essentially everything passed to the CTors to reconstruct the signal
  "Data": [],
  "DsData": None, # No impopsed terms in this example
  "MaxLags": (qx,qy),
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": NonLinearities,
  "MakeRational": MakeRational, 
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ):
    x_val = tor.rand( int( p/3 ) ) # Store since used twice
    x_val -= tor.mean( x_val ) # center
    y_val = Sys(x_val, q, W, Print = False ) # _val to avoid overwriting the training y
    if ( not tor.isnan( tor.sum( y_val ) ) ): break # Remain in the loop until no NaN
  
  ValidationDict["Data"].append( (x_val, y_val) )

# --------------------------------------------------------------------------------------- Fitting ---------------------------------------------------------------------------------------

Arbo = rFOrLSR.Arborescence( y,
                             Ds = RegMat, DsNames = RegNames, # Ds & Regressor names, being dictionary of selected regressors
                             # No Dc or DcNames is passed since we're only imposing 
                             tolRoot = tol, tolRest = tol, # \rho tolerances
                             MaxDepth = 0, # We're only imposing terms, so no arborescence is needed: compute only the root (level 0)
                             ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict, # Validation function and dictionary
                             Verbose = False, # Print the current state of the FOrLSR (only meaningful for regressions with many terms)
                           )

Arbo.fit()

Figs, Axs = Arbo.PlotAndPrint() # returns both figures and axes for further processing
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

plt.show()

print ("\nEnd")