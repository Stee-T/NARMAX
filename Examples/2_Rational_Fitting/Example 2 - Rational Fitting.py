import torch as tor

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import rFOrLSR_0_14 as rFOrLSR
import rFOrLSR_0_14.Test as Test_Systems


p = 2_500 # Dataset size
Amplitude = 1
tol = 0.0001 # Fitting tolerance in %, so multiply by 100
W = None # Gaussian white noise
qx = 3; qy = 3 # maximum x and y delays
ExpansionOrder = 3 # Monomial expansion order
q = max( ( qx, qy ) ) # take the maximum lag to truncate the signals
ArboDepth = 5 # maximum arborescence depth

Sys = Test_Systems.RatNonLinSystem # Example 2 in paper

# ################# Reproducibility #################
import numpy as np
Seed = 85765
print ( "\nSeed: ", Seed, "\n" )
RNG = np.random.default_rng( seed = Seed )
x = tor.Tensor ( ( 2 * Amplitude ) * ( RNG.random( p ) - 0.5 ) ) # uniformly distributed white noise
x -= tor.mean( x ) # center
x, y, W = Sys( x, W, Print = True ) # apply selected system
if ( tor.isnan( tor.sum( y ) ) ): raise AssertionError( "Somehow yields NaNs, Bruh" )
# ################# Reproducability #################


# ---------------------------------------------------------------------------------------  Non-Linearities ---------------------------------------------------------------------------------------
NonLinearities = [ rFOrLSR.Identity ] # List of NonLinearity objects, must start with identity
MakeRational = [ True ] # List of bool, deciding if each NonLinearity is to also be made rational

NonLinearities.append( rFOrLSR.NonLinearity( "abs", f = tor.abs ) )
MakeRational.append( True )


# ---------------------------------------------------------------------------------------  Training data ---------------------------------------------------------------------------------------

y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x, y ), MaxLags = ( qx, qy ) ) # Create the delayed signal terms (cut to q to only have swung-in system)
RegMat, RegNames = rFOrLSR.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = rFOrLSR.CTors.NonLinearizer( y, RegMat, RegNames, NonLinearities, MakeRational = MakeRational ) # add the listed terms to the Regression matrix

print( "\nTolerance: " + str(tol), "\nArboDepth:", ArboDepth, "\nDict-shape:", RegMat.shape, "\nqx =", qx, "; qy =", qy, "\nExpansionOrder =", ExpansionOrder, "\nRationalTerms:", MakeRational[0], "\n\n" )


# --------------------------------------------------------------------------------------- Validation data ---------------------------------------------------------------------------------------

ValidationDict = { # contains essentially everything that is passed to the RegressorMatrix and RegressorTransform() functions
  "Data": [],
  "DsData": None, # No impopsed terms
  "MaxLags": ( qx, qy ),
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": NonLinearities,
  "MakeRational": MakeRational,
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ):
    x_val = Amplitude * tor.rand( int( p / 3 ) ) # Store since used twice
    x_val -= tor.mean( x_val ) # center
    y_val = Sys(x_val, W, Print = False ) # _val to avoid overwriting the training y
    if ( not tor.isnan( tor.sum( y_val ) ) ): break # Remain in the loop until no NaN
  
  ValidationDict["Data"].append( (x_val, y_val) )

# --------------------------------------------------------------------------------------- Fitting ---------------------------------------------------------------------------------------

File = "C:/Users/Stéphane/Desktop/Test.pkl" # Jack

Arbo = rFOrLSR.Arborescence( y,
                             Ds = None, DsNames = None, # Dictionary of selected regressors
                             Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                             tolRoot = tol, tolRest = tol, # \rho tolerances
                             MaxDepth = ArboDepth, # Maximal arborescenc depth
                             ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict, # Validatino function and dictionary
                             Verbose = False, # Print the current state of the FOrLSR (only meaningful for regressions with many terms)
                             # FileName = File, # Path and File to save the Backups into
                             # SaveFrequency = 3, # Save frequency in minutes
                           )

Arbo.fit()

# Arbo = rFOrLSR.Arborescence() # init empty Arbo
# Arbo.load( File )
# Arbo.fit()

Figs, Axs = Arbo.PlotAndPrint() # returns the figures and axes for further processing
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in



################################################################################################### End #########################################################################################
plt.show()
print( "\nEnd" )