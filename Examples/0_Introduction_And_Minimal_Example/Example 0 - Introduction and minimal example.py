import numpy as np
import torch as tor
import NARMAX # of course!

# A) Generate the friend's data
tor.manual_seed( 42 ) # for reproducibility
NumSamples = 250
x, y, z = [ tor.rand( NumSamples, 1 ) for _ in range( 3 ) ] # 1D tor.Tensor

# B) Generate all possible terms
MaxDegree: int = 5
AllMonomials: list[ tor.Tensor ] = []
RegressorNames: list[ str ] = []

for var, name in zip( [ x, y, z ], [ "x", "y", "z" ] ):
  for power in range( 1, MaxDegree + 1 ):
    AllMonomials.append( var**power )
    RegressorNames.append( f"{ name }^{ power }" )

# C) Simulate our friend guess by randomly designing a polynomial (hardcoded to keep the example minimal)
Evaluation: tor.Tensor = 0.3 * x - 0.8 * y**4 + 0.4 * z**3 - 0.5 * x**3 + 0.9 * y**2

# D) Use the library!
Arbo = NARMAX.Arborescence( Evaluation, # 1D tor.Tensor: Our friend's polynomial
                            Dc = tor.column_stack( AllMonomials ), # 2D tor.Tensor: Dictionary of Candidates (Dc)
                            DcNames = np.array( RegressorNames ), # NDArray: Names of Dc's columns
                          )

theta, L = Arbo.fit()[ : 2 ] # Coefficients and indices

print( "\nRecognized regressors:" )
for i in range( len( L ) ): print( theta[ i ].item(), RegressorNames[ L[ i ] ] )
