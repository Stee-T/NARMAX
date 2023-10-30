import rFOrLSR_0_14 as rFOrLSR
import matplotlib.pyplot as plt

import torch as tor

# ---------------------------------------------------------------------------------------- Inputs creation ------------------------------------------------------------------
# Sigmoids data generation
p = 15_000 # dataset size 
TestRange = 8 # Noise Amplitude for testing
Order = 5 # Sigmoid expansion order
# W = np.random.normal(0, 0.005, p) # Gaussian white noise
W = None # Gaussian white noise
ExpansionOrder = 13


if ( Order == 5 ):
  Amplitude = 2.36 # Noise amplitude for fitting
  tol = 1e-5
  theta_CurrentBest = np.array( [1.0014774448118875, 1.0375855925946942, 0.47172276694897614, 0.6339868385212857 , 0.05277150752475056] )
  L_CurrentBest = [ 0, 1, 2, 3, 6 ] # x1, x2, x4, x6, x8 - zero based so -1

elif ( Order == 6 ):
  Amplitude = 2.51 # Noise amplitude for fitting
  tol = 1e-7
  theta_CurrentBest = np.array( [1.0036023116066501, 0.9569033743048118, 0.8475528276554957, 0.4172179435531736, -0.045456003738330183, 0.014615471034694105] )
  L_CurrentBest = [ 0, 1, 2, 4, 5, 7 ] # x1, x2, x3, x5, x8, x11 - zero based so -1

elif ( Order == 7 ):
  Amplitude = 2.45 # Noise amplitude for fitting
  tol = 1e-9
  theta_CurrentBest = np.array( [1.0009318704223191, 0.9917358439783597, 0.6977529433761086, 0.2737472175882833, 0.19906912193943774, 0.030826645127486597, 0.0005398436552096364] )
  L_CurrentBest = [ 0, 1, 2, 3, 4, 6, 9 ] # x, x2, x3, x4, x6, x7, x10 - zero based so -1

else : raise AssertionError( "Order must be 7, 6 or 5" )


# -----------------------------------------------------------------------------------------------------------------------------------------------------------

x = ( 2 * Amplitude ) * ( tor.rand( p ) - 0.5 )
x -= tor.mean( x )
y = tor.tanh( x ) # apply selected system
if ( tor.isnan( tor.sum( y ) ) ): raise AssertionError( "This yields NaNs" )



# ############################################################################# sign/(1-|goes to zero expression|) #######################################################################

# Training data: Don't use the transformation CTor since terms need special processing
y, RegMat, RegNames = rFOrLSR.RegressorMatrix( Data = ( x, y), MaxLags = ( 0, 0 ), ExpansionOrder = ExpansionOrder ) # non-Transformed Regressors (q's +1 to also contain wrong delay terms for testing)

# Make it rational with modified y (can't use the transformation CTor)
y = - y / tor.sign( RegMat[:, 0] ) + 1 # subtract sign of x (RegAt[:,0]) to impose it in regression and divide by it
RegMat = -y.view(-1,1) *  tor.abs( RegMat ) # multiply by -y[k] to put them in denominator
RegNames = list( RegNames ) # cast to list is necessary since the numpy array limits the number of characters in the string
for col in range( len( RegNames ) ): RegNames[col] = f"|{ RegNames[col] }|"
RegNames = np.array( RegNames ) # recast to use multiple indexing

# Center the whole
RegMat -= tor.mean( RegMat, axis = 0, keepdims = True )
y -= tor.mean( y )

# --------------------------------------------------------------------------------------- Validation data ---------------------------------------------------------------------------------------
xValidation = tor.linspace( -TestRange, TestRange, 50_000 )
yValidation = tor.tanh( xValidation )

ValidationDict = { # contains essentially everything that is passed to the RegressorMatrix and RegressorTransform() functions
  "Data": [ ( xValidation, yValidation ) ], # x and y vals are linspaces to just compute the function
  "DsData": None, # No impopsed terms
  "MaxLags": ( 0, 0 ),
  "ExpansionOrder": ExpansionOrder,
  "nC": RegMat.shape[1]
}


def L_inf_Cost( theta, L, ERR, V, MorphDict ): # penalize the maximum deviation

  # --------------------------------------------------------------------  bullshit prevention with readable errors for the user -----------------------------------------------------------------------
  if ( not isinstance( V, dict ) ): raise AssertionError( "The passed V datastructure is not a dictionary as expected" )
  
  # for var in ["Data", "DsData", "MaxLags", "ExpansionOrder", "fPtrs", "fStrs", "nC"]:
  for var in ["Data", "DsData", "MaxLags", "ExpansionOrder", "nC"]: # no functions passed since done manually
    if ( var not in V.keys() ): raise AssertionError( f"The validation datastructure contains no '{var}' entry" )
  
  if ( not isinstance( V["Data"], list ) ):  raise AssertionError( "V's 'Data' entry is expected to be a list" )
  if ( not isinstance( V["nC"], int ) ):     raise AssertionError( "V's 'nC' entry is expected to be an int" )

  Error = 0 # total relative error
  qx = V["MaxLags"][0]; qy = V["MaxLags"][1]; q = max( ( qx, qy ) ) # all swung in states ( x, y, max( x,y ) )
  
  for Sig in range( len( V["Data"] ) ): # iterate over all passed Data tuples
    # --------------------------------------------------------------------  Create Data -----------------------------------------------------------------------
    y, RegMat, RegNames = rFOrLSR.RegressorMatrix( Data = ( V["Data"][Sig][0], V["Data"][Sig][1] ), MaxLags = ( qx,qy ), ExpansionOrder = V["ExpansionOrder"] ) # non-Transformed Regressors
    # RegMat, RegNames, Means, _,  = FOrLSR.RegressorTransform( y, RegMat, RegNames, V["fPtrs"], V["fStrs"], V["MakeRational"] ) # add the listed terms to the Regression matrix

    # --------------------------------------------------------------------  Validation -----------------------------------------------------------------------
    A = tor.abs( RegMat[ :, L.astype( np.int64 ) ] ) @ tor.Tensor( theta ) # A abs-polynomial in paper
    yHat = tor.sign( V["Data"][Sig][0] ) * ( 1 - 1 / (1 + A) ) # create Sigmoid
    
    Error += tor.max( tor.abs( y - yHat ) ) # maximum error
    
  return ( Error / len( V["Data"] ) ) # norm by the number of signals ( not really necessary for AOrLSR but printed for the user )

# --------------------------------------------------------------------------------------- Fitting  ---------------------------------------------------------------------------------------

# Here we exclude the bias in both the fitting and the validation
Arbo = rFOrLSR.Ar( y, # y[k]
                        Ds = None, DsNames = None, # Ds, being dictionary of selected regressors
                        Dc = RegMat, # Dc without y[k], being dictionnary of candidate regerssors (phi)
                        DcNames = RegNames, # Dc's column names
                        tol1 = tol, tol2 = tol, # \rho tolerances
                        MaxDepth = 3, # Maximal arborescence depth
                        F = L_inf_Cost, V = ValidationDict, # Validation function and dictionary
                        verbose = False # Print the current state of the FOrLSR (only meaningful for regressions with mnay terms)
                      ) 

theta, L, ERR, _, _ = Arbo.fit()

for i in range( len( L ) ): print( theta[i], RegNames[ L[i] ].replace("[k]", ""), " + ", end = "" )


# Comparison plot
x_Plot = tor.linspace( -TestRange, TestRange, 5_000 )
# create a RegMat of the regressor matrix and transform it as necessary and use L to acces the corerct terms to recreate the D_x polynomial
RegMat_Plot = FOrLSR.RegressorMatrix( Data = ( x_Plot, x_Plot ), MaxLags = ( 0, 0 ), ExpansionOrder = ExpansionOrder )[1] # non-Transformed Regressors
A = ( tor.abs( RegMat_Plot[ :, L ] ) @ tor.Tensor( theta ) ).cpu()
A_CurrentBest = ( tor.abs( RegMat_Plot[ :, L_CurrentBest ] ) @ tor.Tensor( theta_CurrentBest ) ).cpu()
x_Plot = x_Plot.cpu()

Fig, Ax = plt.subplots()
Ax.plot( x_Plot, tor.tanh( x_Plot ) - tor.sign( x_Plot ) * ( 1 - 1 / (1 + A) ) ) # Difference
Ax.plot( x_Plot, tor.tanh( x_Plot ) - tor.sign( x_Plot ) * ( 1 - 1 / (1 + A_CurrentBest) ) ) # Difference
# Ax.plot( x_Plot, tor.sign( x_Plot ) * ( 1 - 1 / (1 + Z) ) ) # Generated Sigmoid
Ax.grid( which = 'both' )
Ax.legend( ["This Run", "Current Best"] )
Fig.tight_layout()

plt.show()