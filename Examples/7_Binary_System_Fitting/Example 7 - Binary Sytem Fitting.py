# ---------------------------------------------------- 1. Imports
import torch as tor
import numpy as np

import matplotlib.pyplot as plt

plt.style.use( 'dark_background' ) # black graphs <3
import NARMAX
import NARMAX.Test as Test_Systems

# ---------------------------------------------------- 2. Hyper-parameters
p = 2_000 # Dataset size
tol = 0.0001 # Fitting tolerance in %, so multiply by 100
W = None # Noise
ArboDepth = 3 # maximum number of levels the arborescence can have


# ---------------------------------------------------- 3. Training Data
while ( 5 ): # 5 is the absolute truth, do while y contains no nan

  # random binary sequences
  x1 = tor.randint( 0, 2, size = (p,), device = NARMAX.device, dtype = tor.bool )
  x2 = tor.randint( 0, 2, size = (p,), device = NARMAX.device, dtype = tor.bool )
  x3 = tor.randint( 0, 2, size = (p,), device = NARMAX.device, dtype = tor.bool )
  x4 = tor.randint( 0, 2, size = (p,), device = NARMAX.device, dtype = tor.bool )

  x1, x2, x3, x4, y, W = Test_Systems.Binary_MISO_System( x1, x2, x3, x4, W, Print = False )
  if ( not tor.isnan( tor.sum( y ) ) ): break

VarNames = [ "x1", "x2", "x3", "x4" ]
Lags = [ 2, 2, 2, 2 ]
y, RegMat, RegNames = NARMAX.CTors.Lagger( [ x1, x2, x3, x4, y ], Lags + [ 0 ], VarNames + [ "y" ] )
RegMat, RegNames = NARMAX.CTors.Booler( RegMat, RegNames )


# ---------------------------------------------------- 4. Validation Data
ValidationDict = { # contains essentially everything passed to the CTors to reconstruct the regressors
  "y": [],
  "Data": [],
  "InputVarNames": VarNames, # variables in Data, etc
  "Lags": Lags,
}

for i in range( 5 ): # Fill the validation dict's data entry with randomly generated validation data
  x1_val = tor.randint( 0, 2, size = ( int( p / 2 ), ), device = NARMAX.device, dtype = tor.bool )
  x2_val = tor.randint( 0, 2, size = ( int( p / 2 ), ), device = NARMAX.device, dtype = tor.bool )
  x3_val = tor.randint( 0, 2, size = ( int( p / 2 ), ), device = NARMAX.device, dtype = tor.bool )
  x4_val = tor.randint( 0, 2, size = ( int( p / 2 ), ), device = NARMAX.device, dtype = tor.bool )

  _, _, _, _, y_val, W = NARMAX.Test.Binary_MISO_System( x1_val, x2_val, x3_val, x4_val, W, Print = False )
  
  ValidationDict["y"].append( y_val.to( tor.float64 ) )
  ValidationDict["Data"].append( [ x1_val.to( tor.float64 ),
                                   x2_val.to( tor.float64 ),
                                   x3_val.to( tor.float64 ),
                                   x4_val.to( tor.float64 ) ] ) # must be in a list. Only pass input variables


# ---------------------------------------------------- 4. Validation function
def Bool_MAE( theta, L, ERR, RegNames, ValDic,  DcFilterIdx = None ): # The arborescence imposes 6 arguments to passed validation functions

  # ----------------------------------------------- Defensive programming ----------------------------------------------
  if ( not isinstance( ValDic, dict ) ): raise AssertionError( "The passed ValDic datastructure is not a dictionary as expected" )
  
  for var in [ "y", "Data", "InputVarNames", "Lags" ]: # no functions passed since done manually
    if ( var not in ValDic.keys() ): raise AssertionError( f"The validation datastructure contains no '{ var }' entry" )
  
  if ( not isinstance( ValDic["Data"], list ) ):  raise AssertionError( "ValDic's 'Data' entry is expected to be a list" )

  # ---------------------------------------------------- Validation ----------------------------------------------------
  Error = 0 # total error
  
  for i in range( len( ValDic["Data"] ) ): # iterate over all passed Data tuples
    _, RegMat, RegNames = NARMAX.CTors.Lagger( ValDic["Data"][i], ValDic["Lags"], ValDic["InputVarNames"] )
    RegMat, _ = NARMAX.CTors.Booler( RegMat, RegNames ) # create data

    if ( DcFilterIdx is not None ): RegMat = RegMat[:, DcFilterIdx] # Filter out same regressors as for the regression    
    yHat = RegMat[:, L.astype( np.int64 ) ].to( tor.float64 ) @ theta # legal since MA system

    Error += tor.mean( tor.abs( ValDic["y"][i] - yHat ) / tor.mean( tor.abs( ValDic["y"][i] ) ) ).item() # relative MAE
    
  return ( Error / len( ValDic["Data"] ) ) # norm by the number of validation-sets ( not really necessary for AOrLSR but printed for the user )


# ---------------------------------------------------- 5. Running the Arborescence
Arbo = NARMAX.Arborescence( y.to( tor.float64 ),
                            Dc = RegMat.to( tor.float64 ), DcNames = RegNames, # Dc & Regressor names, being dictionnary of candidate regerssors (phi)
                            tolRoot = tol, tolRest = tol, # \rho tolerances
                            MaxDepth = ArboDepth, # Maximal number of levels
                            ValFunc = Bool_MAE, ValData = ValidationDict, # Validation function and dictionary
                          )

theta, L, ERR, _, RegMat, RegNames = Arbo.fit()


# ---------------------------------------------------- 6. Result Analysis
print( "System: y[k] = ( !x1[k] && x2[k] ) - ( x3[k] ^ x4[k] ) + ( x1[k] || x3[k-1] ) "
                                "+ ( x2[k] ^ x4[k-2] ) - ( !x1[k-2] && x2[k] ) - !x3[k-2] + !x2[k-2] + x4[k-1] \n\nAOrLSR Results:")
for i in range( len( theta ) ): print( theta[i], RegNames[ L[i] ] )

Fig, Ax = plt.subplots()
Ax.plot( y.cpu().numpy() - RegMat[:, L.astype( np.int64 ) ].to( tor.float64 ).cpu().numpy() @ theta, marker = '.', markersize = 6 )
Ax.grid( which = 'both' )
Ax.legend( [ "y - yHat" ] )
Fig.tight_layout()

plt.show()

print( "\nend" )