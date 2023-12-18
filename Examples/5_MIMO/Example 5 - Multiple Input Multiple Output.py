import rFOrLSR_0_14 as rFOrLSR
import rFOrLSR_0_14.Test as Test_Systems

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import torch as tor

# ################################################################################ Inputs creation ################################################################################
# Sigmoids data generation
p = 2_000 # dataset size 
ExpansionOrder = 3 # allow up to x^3
InputAmplitude = 1.5 # Set Noise amplitude
tol = 1e-9 # Desired tolerance
ArboDepth = 4 # maximum number of levels the arborescence can have
W = None
Lags = ( 3, 3, 3, [1, 2, 3], [1, 2, 3] ) # x1, x2, x3, y1, y2. Exclude y1[k] and y2[k]

# Generate x and y data
while ( 5 ): # 5 is the absolute truth, do while y contains no nan
  x1 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
  x2 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
  x3 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )

  x1 -= tor.mean( x1 ) # important as always
  x2 -= tor.mean( x2 ) # important as always
  x3 -= tor.mean( x3 ) # important as always
  
  x1, x2, x3, y1, y2, W = Test_Systems.ThreeInputMIMO( x1, x2, x3, W, Print = True ) # apply selected system
  if ( not tor.isnan( tor.sum( y1 ) ) and not tor.isnan( tor.sum( y2 ) ) ): break


NonLinearities = [ rFOrLSR.Identity, rFOrLSR.NonLinearity( "abs", f = tor.abs ) ] # List of NonLinearity objects, must start with identity

# ---------------------------------------------------- 3. Training Data
_, RegMat, RegNames = rFOrLSR.CTors.Lagger( Data = [ x1, x2, x3, y1, y2 ], Lags = Lags, RegNames = [ "x1", "x2", "x3", "y1", "y2" ] ) # Create the delayed regressors
RegMat, RegNames = rFOrLSR.CTors.Expander( RegMat, RegNames, ExpansionOrder ) # Monomial expand the regressors
RegMat, RegNames, _ = rFOrLSR.CTors.NonLinearizer( None, RegMat, RegNames, NonLinearities ) # add the listed terms to the Regression matrix

# Cut y1, y2 to the same length as RegMat
y1 = rFOrLSR.CutY( y1, Lags )
y2 = rFOrLSR.CutY( y2, Lags )

# ---------------------------------------------------- 4. Validation Data
ValidationDict1 = { # contains essentially everything passed to the CTors to reconstruct the regressor
  "y": [],
  "Data": [],
  "InputVarNames": [ "x1", "x2", "x3", "y1", "y2" ], # RegNames for Lagger
  "DsData": None, # No impopsed terms in this example
  "Lags": Lags,
  "ExpansionOrder": ExpansionOrder,
  "NonLinearities": NonLinearities,
  "MakeRational": None, 
}

import copy

ValidationDict2 = copy.deepcopy( ValidationDict1 )

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  while ( 5 ): # 5 is the absolute truth

    x1_val = tor.rand( int( p / 3 ) ) # Store since used twice
    x2_val = tor.rand( int( p / 3 ) ) # Store since used twice
    x3_val = tor.rand( int( p / 3 ) ) # Store since used twice

    x1_val -= tor.mean( x1_val ) # Center
    x2_val -= tor.mean( x2_val ) # Center
    x3_val -= tor.mean( x3_val ) # Center

    x1_val, x2_val, x3_val, y1_val, y2_val, W = Test_Systems.ThreeInputMIMO( x1_val, x2_val, x3_val, W, Print = False ) # _val to avoid overwriting the training y

    if ( not tor.isnan( tor.sum( y1_val ) ) and not tor.isnan( tor.sum( y2_val ) ) ): break # Remain in the loop until no NaN
  
  # Each system output needs it final output
  ValidationDict1["y"].append( rFOrLSR.CutY( y1_val, ValidationDict1["Lags"] ) ) # Cuts the y to the right size (avois a warning)
  ValidationDict2["y"].append( rFOrLSR.CutY( y2_val, ValidationDict2["Lags"] ) ) # Cuts the y to the right size (avois a warning)
  
  # Both system output require the full data
  ValidationDict1["Data"].append( [ x1_val, x2_val, x3_val, y1_val, y2_val ] ) # uncut y
  ValidationDict2["Data"].append( [ x1_val, x2_val, x3_val, y1_val, y2_val ] ) # uncut y


# ---------------------------------------------------- 5. Running the Arborescences
Arbo_1 = rFOrLSR.Arborescence( y1,
                               Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                               tolRoot = tol, tolRest = tol, # \rho tolerances
                               MaxDepth = ArboDepth, # Maximal number of levels
                               ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict1, # Validation function and dictionary
                              )
Arbo_1.fit() # Don't overwrite RegMat, since it is used in the next Arbo


Arbo_2 = rFOrLSR.Arborescence( y2,
                               Dc = RegMat, DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                               tolRoot = tol, tolRest = tol, # \rho tolerances
                               MaxDepth = ArboDepth, # Maximal number of levels
                               ValFunc = rFOrLSR.DefaultValidation, ValData = ValidationDict2, # Validation function and dictionary
                              )
Arbo_2.fit()


Figs, Axs = Arbo_1.PlotAndPrint() # returns both figures and axes for further processing, as as the zoom-in below
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in

Figs, Axs = Arbo_2.PlotAndPrint() # returns both figures and axes for further processing, as as the zoom-in below
Axs[0][0].set_xlim( [0, 500] ) # Force a zoom-in


plt.show()