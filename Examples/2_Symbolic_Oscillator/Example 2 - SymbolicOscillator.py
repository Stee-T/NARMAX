import torch as tor
import matplotlib.pyplot as plt
import NARMAX

# --------------------------------------------------- 1. Data Generation
p: int = 2_000 # dataset size
InputAmplitude: float = 1.5

x1: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 ) # Mean not subtracted, since no rFOrLSR
x2: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )

def UserFunction( x: tor.Tensor ) -> tor.Tensor:
  return ( tor.max( tor.tensor( [ 0 ] ), tor.sin( 2 * x ) ) ) # whatever really

def System( y: tor.Tensor, x1: tor.Tensor, x2: tor.Tensor, W: tor.Tensor, theta: tor.Tensor,
            StartIdx: int, EndIdx: int ) -> tor.Tensor:
  for k in range( StartIdx, EndIdx ):
    y[k] = W[k] + ( ( theta[0] * y[k-1] / x2[k] + theta[1] * UserFunction( x1[k-1] )
                      + theta[2] / tor.abs( 0.2 * x1[k-1] + 0.5 * x1[k-2] * x2[k] - 0.2 ) ) # Numerator
                                                    /
                    ( 1 + theta[3] * x1[k-1] * x2[k-1] + theta[4] * x2[k-2]**2
                      + theta[5] * tor.cos( 0.2 * x1[k-3] * x2[k-1] - 0.1 ) ) ) # Denominator
  return ( y )

# Direct input to the system, allows to model input noise / supplementary regressors, DC-offset or whatever
AdditionalInput: tor.Tensor = 0.2 * ( tor.rand( p ) - 0.5 )

# System / regression coefficients, change to emulate modulation
theta: list[ tor.Tensor ] = [ tor.tensor( [ 0.2,  -0.3,  1,   0.8, -0.3, 1 ] ),
                              tor.tensor( [ 0.25, -0.25, 0.8, 0.9, -0.5, 0.95 ] ),
                              tor.tensor( [ 0.3,  -0.4,  0.7, 0.7, -0.4, 0.9 ] )
                            ] 

ThirdBuffer: int = p // 3 # How long we keep the same theta values

# --------------------------------------------------- 2. Symbolic Oscillator Data
# Declare what we need to emulate the system
InputVarNames: list[ str ] = [ 'x1', 'x2', 'y' ] # Used Variables
NonLinearities: list[ NARMAX.NonLinearity ] = [ NARMAX.Identity, # Obligatory for AOrLSR, here optional
                                                NARMAX.NonLinearity( "uFunc", f = UserFunction ),
                                                NARMAX.NonLinearity( "abs", f = tor.abs ),
                                                NARMAX.NonLinearity( "cos", f = tor.cos ),
                                              ] # Used Functions

Expressions: list[ str ] = ["y[k-1]/x2[k]", "uFunc( x1[k-1] )", "1/abs( 0.2*x1[k-1] + 0.5*x1[k-2]*x2[k] - 0.2 )", # Numerator
                            "~/(x1[k-1]*x2[k-1])", "~/(x2[k-2]^2)", "~/cos( 0.2*x1[k-3]*x2[k-1] - 0.1 )" ] # Denominator

# ----------------------------------------------------- 3. Simulation and Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ y = Real System
y: tor.Tensor = tor.zeros( p )
for i in range( 3 ):
  StartIdx: int = ( i * ThirdBuffer ) if ( i > 0 ) else 3 # avoids index error in System
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  y: tor.Tensor = System( y, x1, x2, AdditionalInput, theta[i], StartIdx, EndIdx ) # only overwrite y[StartIdx:EndIdx]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ yHat = Symbolic Oscillator
yHat: tor.Tensor = tor.zeros( p )
Model: NARMAX.SymbolicOscillator = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[0] )

for i in range( 3 ):
  StartIdx: int = i * ThirdBuffer # no conditional, since SymbOsc-class handles buffer starts correctly
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  yHat[ StartIdx : EndIdx ] = Model.Oscillate( Data = [ x1[ StartIdx : EndIdx], x2[ StartIdx : EndIdx ] ],
                                               theta = theta[i], # change regression coefficients
                                               DsData = AdditionalInput[ StartIdx : EndIdx ] # additional input
                                             )
  # Model.set_theta( theta[i] ) # would also work if separating the system update from the processing is desired

# ------------------------------------------------- 4. Error Analysis
Fig, Ax = plt.subplots()
Diff: tor.Tensor = (y - yHat)[20:] # cut the start since the system init of the foor loops is incomplete
Ax.plot( Diff.cpu().numpy() ) 
Ax.set( title = "y - yHat", xlabel = 'k', ylabel = 'y - yHat',
        xlim = ( 0, p-20 ), ylim = ( 1.1*Diff.min().item(), 1.1*Diff.max().item() )
      )
Ax.grid( which = 'both', alpha = 0.3 )
Fig.tight_layout()

plt.show()