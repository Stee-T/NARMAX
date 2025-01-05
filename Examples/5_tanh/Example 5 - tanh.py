import NARMAX

import matplotlib.pyplot as plt
plt.style.use( 'dark_background' ) # black graphs <3

import numpy as np
import torch as tor

# ################################################################################ Inputs creation ################################################################################
# Sigmoids data generation
p = 50_000 # dataset size 
TestRange = 8 # Noise Amplitude for testing
ExpansionOrder = 13 # allow up to x^13
Amplitude = 2.45 # Set Noise amplitude
tol = 1e-9 # Desired tolerance

x = ( 2 * Amplitude ) * ( tor.rand( p ) - 0.5 )
x -= tor.mean( x ) # important as always
y = tor.tanh( x ) # desired function

# ############################################################################ Training Data creation ###########################################################################
RegMat, RegNames = NARMAX.CTors.Expander( Data = tor.abs( x.view( -1, 1 ) ), RegNames = [ "|x|" ], ExpansionOrder = ExpansionOrder )
y = - y / tor.sign( x ) + 1 # subtract sign(x) to impose it in regression and divide by it
RegMat = -y.view( -1, 1 ) * RegMat # multiply by -y to put regressors in denominator


# ############################################################################# Validation handling ###########################################################################
xValidation = tor.linspace( -TestRange, TestRange, 50_000 ) # high number of samples for precise max deviation search
yValidation = tor.tanh( xValidation )

ValidationDict = { # Contains only the data used by our custom validation function
  "y": [ yValidation ],
  "Data": [ xValidation.view( -1, 1 ) ], # x is a linspaces to just compute the function
  "ExpansionOrder": ExpansionOrder,
}

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


# ######################################################################### Fitting #########################################################################

Arbo = NARMAX.Arborescence( y,
                            Dc = RegMat, DcNames = RegNames, # dictionary of candidates and the column names
                            tolRoot = tol, tolRest = tol, # \rho tolerances
                            MaxDepth = 3, # Maximal arborescence depth
                            ValFunc = Sigmoid_Expansion_L_inf, ValData = ValidationDict, # Validation function and dictionary
                          )

theta, L, ERR, _, RegMat, RegNames = Arbo.fit()

Expression = ""
for i in range( len( theta ) ): Expression += f"{ theta[i] } { RegNames[ L[i] ] } + "

print( "\n", Expression[ : -3 ], "\n" )


# Error plot
x_Plot = tor.linspace( -TestRange, TestRange, 5_000 )

# create a RegMat of the regressor matrix and transform it as necessary and use L to acces the corerct terms to recreate the A polynomial
RegMat_Plot = NARMAX.CTors.Expander( Data = x_Plot.view( -1, 1 ), RegNames = [ "x" ], ExpansionOrder = ExpansionOrder )[0] # ignore the names

A = ( tor.abs( RegMat_Plot[ :, L ] ) @ tor.tensor( theta ) ).cpu()
x_Plot = x_Plot.cpu()

Fig, Ax = plt.subplots()
Ax.plot( x_Plot, tor.tanh( x_Plot ) - tor.sign( x_Plot ) * ( 1 - 1 / (1 + A) ) ) # Plot the Error
Ax.grid( which = 'both', alpha = 0.5 )
Ax.legend( ["Fitting error"] )
Fig.tight_layout()

plt.show()

# TODO: maka a NARMAX object for this, good test with absolutely memory-less sytem