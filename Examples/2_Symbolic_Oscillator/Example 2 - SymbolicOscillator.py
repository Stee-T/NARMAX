import torch as tor
import matplotlib.pyplot as plt
import rFOrLSR

# --------------------------------------------------- 3. Data Generation
p = 2_000 # dataset size 
InputAmplitude = 1.5 # Set Noise amplitude

x1 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 ) # Mean not subtracted, since no rFOrLSR
x2 = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )

def System( y, x1, x2, W, theta, StartIdx, EndIdx ):
  for k in range( StartIdx, EndIdx ): # start at 2 since the biggest negative lag is k-2
    y[k] = W[k] + ( ( theta[0]*y[k-1]/x2[k] + theta[1]*x1[k-1]**2 + theta[2]/tor.abs( 0.2*x1[k-1] + 0.5*x1[k-2]*x2[k] - 0.2 ) ) / # Numerator
                    ( 1 + theta[3]*x1[k-1]*x2[k-1] + theta[4]*x2[k-2]**2 + theta[5]*tor.cos( 0.2*x1[k-3]*x2[k-1] - 0.1 ) )    ) # Denominator
  return ( y )

# Declare what we used to emulate
Data = [ x1, x2 ] # Used Input Data
InputVarNames = [ 'x1', 'x2', 'y' ] # Used Variables
NonLinearities = [ rFOrLSR.Identity, rFOrLSR.NonLinearity( "abs", f = tor.abs ), rFOrLSR.NonLinearity( "cos", f = tor.cos ) ] # Used Functions
Expressions = ["y[k-1]/x2[k]", "x1[k-1]^2", "1/abs( 0.2*x1[k-1] + 0.5*x1[k-2]*x2[k] - 0.2 )", # Num
                "~/(x1[k-1]*x2[k-1])", "~/(x2[k-2]^2)", "~/cos( 0.2*x1[k-3]*x2[k-1] - 0.1 )" ] # Used Regressors (1 in deno is implicit)

ThirdBuffer = int( p / 3 )

# Direct input to the system, allows to model input noise / supplementary regressors, DC-offset or whatever
AdditionalInput = 0.2 * ( tor.rand( p ) - 0.5 )

# System / regression coefficients, change to emulate modulation
theta = [ tor.tensor( [ 0.2,  -0.3,  1,   0.8, -0.3, 1 ] ),
          tor.tensor( [ 0.25, -0.25, 0.8, 0.9, -0.5, 0.95 ] ),
          tor.tensor( [ 0.3,  -0.4,  0.7, 0.7, -0.4, 0.9 ] )
        ] 

# ----------------------------------------------------- 2. Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ y = Real System
y = tor.zeros( p )
for i in range( 3 ):
  StartIdx = (i * ThirdBuffer) if ( i > 0 ) else 3 # avoids index error in System
  EndIdx = (i+1) * ThirdBuffer
  y = System( y, x1, x2, AdditionalInput, theta[i], StartIdx, EndIdx )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ yHat = Symbolic Oscillator
yHat = tor.zeros( p )
NARMAX = rFOrLSR.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[0] )

for i in range( 3 ):
  StartIdx = i * ThirdBuffer # no conditional, since NARMAX-class handles buffer starts correctly
  EndIdx = (i+1) * ThirdBuffer
  yHat[ StartIdx : EndIdx] = NARMAX.Oscillate( Data = [ x1[ StartIdx : EndIdx], x2[ StartIdx : EndIdx] ],
                                               theta = theta[i], # change regression coefficients
                                               DsData = AdditionalInput[ StartIdx : EndIdx] # additional input
                                             )
  # NARMAX.set_theta( theta[i] ) # would also work if separating the system update from the processing is desired


# ------------------------------------------------- 3. Testing
Fig, Ax = plt.subplots()
Diff = (y - yHat)[20:] # cut the start since the system init of the foor loops is incomplete
Ax.plot( Diff.cpu().numpy() ) 
Ax.set( title = "y - yHat", xlabel = 'k', ylabel = 'y - yHat',
        xlim = ( 0, p-20 ), ylim = ( 1.1*Diff.min().item(), 1.1*Diff.max().item() )
      )
Ax.grid( which = 'both', alpha = 0.3 )
Fig.tight_layout()

plt.show()