# ---------------------------------------------------- 1. Imports
from typing import Any, Optional
import torch as tor

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import NARMAX
import NARMAX.Test as Test_Systems

# ---------------------------------------------------- 2. Hyper-parameters
p: int = 2_500 # Dataset size
InputAmplitude: float = 1.0
tol: float = 0.0001 # Fitting tolerance in %, so multiply by 100
W: Optional[tor.Tensor] = None # Noise
qx: int = 4; qy: int = 4 # maximum x and y delays
ExpansionOrder: int = 3 # monomial expansion order
ArboDepth: int = 4 # maximum number of levels the arborescence can have

# System choice
# Sys = Test_Systems.ARX
# Sys = Test_Systems.TermCombinations
# Sys = Test_Systems.iFOrLSR
Sys = Test_Systems.NonLinearities
# Sys = Test_Systems.NotInDictionaty

# Generate x and y data
while ( 5 ): # 5 is the absolute truth, do while y contains no nan
  x: tor.Tensor = ( InputAmplitude * 2 ) * ( tor.rand( p, device = NARMAX.Device ) - 0.5 ) # uniformly distributed white noise
  x -= tor.mean( x ) # center: VERY IMPORTANT!
  x, y, W = Sys( x, W, Print = True ) # apply selected system
  if ( not tor.isnan( tor.sum( y ) ) ): break


NonLinearities: list[ NARMAX.NonLinearity ] = [ NARMAX.Identity ] # List of NonLinearity objects, must start with identity
NonLinearities.append( NARMAX.NonLinearity( "abs", f = tor.abs ) )
NonLinearities.append( NARMAX.NonLinearity( "cos", f = tor.cos ) )
NonLinearities.append( NARMAX.NonLinearity( "exp", f = tor.exp ) )

# ---------------------------------------------------- 3. Training Data
y, RegMat, RegNames = NARMAX.CTors.Lagger( Data = ( x, y ), Lags = ( qx, qy ) ) # Create the delayed regressors
RegMat, RegNames = NARMAX.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = NARMAX.CTors.NonLinearizer( y, RegMat, RegNames, NonLinearities ) # add the listed terms to the Regression matrix


# ---------------------------------------------------- 4. Validation Data
ValidationDict: dict[ str, Any ] = { # contains essentially everything passed to the CTors to reconstruct the regressors
  "y": [],
  "Data": [],
  "InputVarNames": [ "x", "y" ], # variables in Data, etc
  "NonLinearities": NonLinearities,
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ): # 5 is the absolute truth
    x_val: tor.Tensor = tor.rand( int( p / 3 ) ) # Store since used twice
    x_val -= tor.mean( x_val ) # center
    x_val, y_val, W = Sys( x_val, W, Print = False ) # _val to avoid overwriting the training y
    if ( not tor.isnan( tor.sum( y_val ) ) ): break # Remain in the loop until no NaN
  
  ValidationDict["y"].append( y_val )
  ValidationDict["Data"].append( [ x_val ] ) # must be in a list. Only pass input variables


# ---------------------------------------------------- 5. Running the Arborescence
# File = "Some/Valid/Path/FileName.pkl"

Arbo = NARMAX.Arborescence( y,
                            Ds = None, DsNames = None, # Ds & Regressor names, being dictionary of selected regressors
                            Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                            tolRoot = tol, tolRest = tol, # \rho tolerances
                            MaxDepth = ArboDepth, # Maximal number of levels
                            ValFunc = NARMAX.DefaultValidation, ValData = ValidationDict, # Validation function and dictionary
                            Verbose = False, # Print the current state of the FOrLSR (only meaningful for regressions with many terms)
                            # FileName = File, # Path and File to save the Backups into
                            # SaveFrequency = 10, # Save frequency in minutes
                          )

Arbo.fit()

# If the Arborescence was interrupted and saved, continue with:
# Arbo = NARMAX.Arborescence() # init empty Arbo
# Arbo.load( File ) # load pickle file
# Arbo.fit() # resume fitting

Figs, Axs = Arbo.PlotAndPrint( ValidationDict ) # returns both figures and axes for further processing, as as the zoom-in below
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

theta, L, ERR, _, RegMat, RegNames = Arbo.get_Results() # See Tutorial 2 for how use that data!

plt.show()
