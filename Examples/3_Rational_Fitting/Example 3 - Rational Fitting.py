# ---------------------------------------------------- 1. Imports
import torch as tor

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import rFOrLSR
import rFOrLSR.Test as Test_Systems

# ---------------------------------------------------- 2. Hyper-parameters
p = 2_500 # Dataset size
InputAmplitude = 1
tol = 0.0001 # Fitting tolerance in %, so multiply by 100
W = None # Noise
qx = 3; qy = 3 # maximum x and y delays
ExpansionOrder = 3 # monomial expansion order
ArboDepth = 4 # maximum number of levels the arborescence can have

Sys = Test_Systems.RatNonLinSystem # Example 2 in paper

import numpy as np
Seed = 252 # use a fixed seed to reproduce results
print ( "\nSeed: ", Seed, "\n" )
RNG = np.random.default_rng( seed = Seed )
x = tor.tensor( ( 2 * InputAmplitude ) * ( RNG.random( p ) - 0.5 ) ) # uniformly distributed white noise
x -= tor.mean( x ) # center
x, y, W = Sys( x, W, Print = True ) # apply selected system
if ( tor.isnan( tor.sum( y ) ) ): raise AssertionError( "Yields NaNs, which we don't like" )

NonLinearities = [ rFOrLSR.Identity ] # List of NonLinearity objects, must start with identity
MakeRational = [ True ] # List of bool, deciding if each NonLinearity is to also be made rational

NonLinearities.append( rFOrLSR.NonLinearity( "abs", f = tor.abs ) )
MakeRational.append( True )

# ---------------------------------------------------- 3. Training Data
y, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) ) # Create the delayed regressors
RegMat, RegNames = rFOrLSR.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = rFOrLSR.CTors.NonLinearizer( y, RegMat, RegNames, NonLinearities, MakeRational = MakeRational ) # add the listed terms to the Regression matrix


# ---------------------------------------------------- 4. Validation Data
ValidationDict = { # contains essentially everything passed to the CTors to reconstruct the regressors
  "y": [],
  "Data": [],
  "InputVarNames": [ "x", "y" ], # variables used in the system
  "DcNames": RegNames,
  "NonLinearities": NonLinearities,
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ): # 5 is the absolute truth
    x_val = tor.rand( int( p / 3 ) ) # Store since used twice
    x_val -= tor.mean( x_val ) # CENTER!!!
    x_val, y_val, W = Sys( x_val, W, Print = False ) # _val to avoid overwriting the training y
    if ( not tor.isnan( tor.sum( y_val ) ) ): break # Remain in the loop until no NaN
  
  ValidationDict["y"].append( y_val ) # Cuts the y to the right size (avois a warning)
  ValidationDict["Data"].append( [ x_val ] ) # List of only input terms


# ---------------------------------------------------- 5. Running the Arborescence
File = "Some/Valid/Path/FileName.pkl"

Arbo = rFOrLSR.Arborescence( y,
                             Ds = None, DsNames = None, # Ds & Regressor names, being dictionary of selected regressors
                             Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                             tolRoot = tol, tolRest = tol, # \rho tolerances
                             MaxDepth = ArboDepth, # Maximal number of levels
                             ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict, # Validation function and dictionary
                             Verbose = False, # Print the current state of the FOrLSR (only meaningful for regressions with many terms)
                            #  FileName = File, # Path and File to save the Backups into
                            #  SaveFrequency = 10, # Save frequency in minutes
                           )

Arbo.fit()

# If the Arborescence was interrupted and saved, continue with:
# Arbo = rFOrLSR.Arborescence() # init empty Arbo
# Arbo.load( File ) # load pickle file
# Arbo.fit() # resume fitting

Figs, Axs = Arbo.PlotAndPrint( ValidationDict ) # returns both figures and axes for further processing, as as the zoom-in below
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

theta, L, ERR, _, RegMat, RegNames = Arbo.get_Results()

# ---------------------------------------------------- 6. Run the system
TestInput = tor.tensor( ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 ) ) # uniformly distributed white noise
TestInput -= TestInput.mean() # center

_, y, W = Sys( TestInput, W, Print = True ) # Original for-loop

NARMAX = rFOrLSR.SymbolicOscillator( [ "x", "y" ], NonLinearities, RegNames[L], tor.tensor( theta ) )

# Prefill the internal state to perfectly emulate the ground truth y
NARMAX.set_InputStorage( TestInput[ None, :NARMAX.get_MaxNegInputLag() ].clone() ) # corresponds to qx in this example
NARMAX.set_OutputStorage( y[ :NARMAX.get_MaxNegOutputLag() ].clone() ) # corresponds to qy in this example
yHat = NARMAX.Oscillate( [ TestInput[ -len( y ): ] ] ) # generate data

# Note: We cut the input w.r.t. len( y ), since the generated model might have different lags.
# Otherwise max( NARMAX.get_MaxNegOutputLag(), NARMAX.get_MaxNegInputLag() ) is the way to go##

Fig, Ax = plt.subplots()
Ax.plot( (y - yHat )[10:].cpu().numpy(), label = "y - yHat" )
Ax.grid( which = "both", alpha = 0.3 )


plt.show()