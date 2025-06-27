# FOrLSR.Test file
import torch as tor
from typing import Optional, Tuple, Callable

def InputCheck( x: tor.Tensor, W: Optional[ tor.Tensor ], Print: bool ) -> None:
  if ( not isinstance( x, tor.Tensor ) ): raise ValueError( "x must be a torch.Tensor" )
  if ( x.dim() != 1 ): raise ValueError( "x must be a 1D torch.Tensor" )

  if ( W is not None ): 
    if ( ( not isinstance( W, tor.Tensor ) ) or ( W.dim() != x.dim() ) ): raise ValueError( "W must be None or a torch.Tensor of the same dimension as x" )

  if ( not isinstance( Print, bool ) ): raise ValueError( "Print must be a bool" )



# ############################################################################## Nonlinear-MA system ###############################################################################
def MA( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = 0.2*x[k] + 0.7*x[k-1] -0.5*x[k-1]**2 + 0.6*x[k-2] -0.7*x[k-2]**3 + 0.4*x[k-3] -0.3*y[k-3]**2 -0.5*x[k-4] + 0.3*y[k-4]**2
  
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x, W, Print )

  MaxLag: int = 4
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 4
    y[k] = 0.2*x[k] + 0.7*x[k-1] -0.5*x[k-1]**2 + 0.6*x[k-2] -0.7*x[k-2]**3 + 0.4*x[k-3] -0.3*x[k-3]**2 -0.5*x[k-4] + 0.3*x[k-4]**2
    if ( W is not None ): y[k] += W[k] # Additive Noise 
  
  if ( Print ): print("System: y[k] = 0.2*x[k] + 0.7*x[k-1] -0.5*x[k-1]^2 + 0.6*x[k-2] -0.7*x[k-2]^3 + 0.4*x[k-3] -0.3*x[k-3]^2 -0.5*x[k-4] + 0.3*x[k-4]^2 \n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )

# ###################################################################################### ARX #######################################################################################
def ARX( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = 0.2*x[k] + 0.7*x[k-1] + 0.6*x[k-2] +0.4*x[k-3] -0.5*x[k-4] -0.5*y[k-1] -0.7*y[k-2] -0.3*y[k-3] + 0.3*y[k-4]
  
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x, W, Print )

  MaxLag: int = 4
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 4
    y[k] = 0.2*x[k] + 0.7*x[k-1] + 0.6*x[k-2] +0.4*x[k-3] -0.5*x[k-4] -0.5*y[k-1] -0.7*y[k-2] -0.3*y[k-3] + 0.3*y[k-4]
    if ( W is not None ): y[k] += W[k] # Additive Noise 
  
  if ( Print ): print("System: y[k] = 0.2*x[k] + 0.7*x[k-1] + 0.6*x[k-2] +0.4*x[k-3] -0.5*x[k-4] -0.5*y[k-1] -0.7*y[k-2] -0.3*y[k-3] + 0.3*y[k-4] \n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


# ############################################################################### Term Combinations ################################################################################
def TermCombinations( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = -0.2x[k] -0.4x[k-1]^3 + 0.3x[k-2]*x[k-1]^2 + 0.5x[k-3]*x[k-2] + 0.5y[k-1] + 0.2y[k-2]^2 - 0.1y[k-3]^3
    
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
'''

  InputCheck( x, W, Print )

  MaxLag: int = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = -0.2*x[k] - 0.4*x[k-1]**3 + 0.3*x[k-2]*x[k-1]**2 + 0.5*x[k-3]*x[k-2] + 0.5*y[k-1] + 0.2*y[k-2]**2 -0.1*y[k-3]**3
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = -0.2x[k] -0.4x[k-1]^3 + 0.3x[k-2]*x[k-1]^2 + 0.5x[k-3]*x[k-2] + 0.5y[k-1] + 0.2y[k-2]^2 - 0.1y[k-3]^3\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


## ######################################################################################## iFOrLSR ################################################################################
 # iFOrLSR paper example
def iFOrLSR( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = 0.2y[k-1]^3 + 0.7y[k-1]*x[k-1] +0.6x[k-2]^2 -0.5y[k-2] -0.7y[k-2]*x[k-2]^2
    
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x, W, Print )

  MaxLag: int = 2
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 2
    y[k] = 0.2*y[k-1]**3 + 0.7*y[k-1]*x[k-1] + 0.6*x[k-2]**2 -0.5*y[k-2] -0.7*y[k-2]*x[k-2]**2
    if ( W is not None ): y[k] += W[k] # Additive Noise 

  if ( Print ): print("System: y[k] = 0.2y[k-1]^3 + 0.7y[k-1]*x[k-1] +0.6x[k-2]^2 -0.5y[k-2] -0.7y[k-2]*x[k-2]^2\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


# ################################################################################ Non-Linearities #################################################################################
def NonLinearities( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = 0.3x[k] + 0.3x[k-1]^3 + 0.7|x[k-2]*x[k-1]^2| - 0.5exp(x[k-3]*x[k-2]) + 0.5cos(y[k-1]*x[k-2]) - 0.4|x[k-1]*y[k-2]^2| - 0.4y[k-3]^3
  
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x, W, Print )

  MaxLag: int = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = 0.3*x[k] + 0.3*x[k-1]**3 + 0.7*tor.abs( x[k-2]*x[k-1]**2) - 0.5*tor.exp( x[k-3]*x[k-2] ) + 0.5*tor.cos( y[k-1] * x[k-2] ) - 0.4*tor.abs( x[k-1] * y[k-2]**2 ) - 0.4*y[k-3]**3 # AOrLSR benchmark version
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = 0.3x[k] + 0.3x[k-1]^3 + 0.7|x[k-2]*x[k-1]^2| - 0.5exp(x[k-3]*x[k-2]) + 0.5cos(y[k-1]*x[k-2]) - 0.4|x[k-1]*y[k-2]^2| - 0.4y[k-3]^3\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )
  
 
# ################################################################################## Not In Dict ###################################################################################
def SevereNonLinearities( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]: # Model with terms
  '''0.5*cos^3(0.2*x[k] + 0.3*x[k-1]) + 0.2*tan( x[k-2]*x[k-1]) + 0.5*abs(tanh( x[k-3]*x[k-2])) + 0.1*abs^3(y[k-3]-0.1)
    
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''
  
  InputCheck( x, W, Print )

  MaxLag: int = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state
  
  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = 0.5*tor.cos(0.2*x[k] + 0.3*x[k-1])**3 + 0.2*tor.tan( x[k-2]*x[k-1]) + 0.5*tor.abs(tor.tanh( x[k-3]*x[k-2])) + 0.1*tor.abs(y[k-3]-0.1)**3
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: 0.5*cos^3(0.2*x[k] + 0.3*x[k-1]) + 0.2*tan( x[k-2]*x[k-1]) + 0.5*abs(tanh( x[k-3]*x[k-2])) + 0.1*abs^3(y[k-3]-0.1) \n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


# ############################################################################## Rational Non-linear ###############################################################################
# From the AOrLSR paper
def RatNonLinSystem( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]: # Rational system used
  '''y[k] = ( 0.6*abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*abs( y[k-1] )  ) / ( 1 - 0.4*abs( x[k] ) + 0.3*abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] )
    
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x, W, Print )

  MaxLag: int = 2
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 2
    y[k] = ( 0.6*tor.abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*tor.abs( y[k-1] )  ) / ( 1 - 0.4*tor.abs( x[k] ) + 0.3*tor.abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] ) # With mixed terms
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = ( 0.6*abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*abs( y[k-1] )  ) / ( 1 - 0.4*abs( x[k] ) + 0.3*abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] )")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


# ########################################################################## Rational Non-linear System 2 ##########################################################################
def RationalNARMAX( x: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = True ) -> Tuple[tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = \n(2x[k] -0.4|x[k-1]^3| + x[k-2]*x[k-1]^2 + 0.5sqrt(|x[k-3]|) /\n(0.5cos(y[k-1]) + 2y[k-2]^2)
    
  ### Inputs:
  - `x`: ((p,)-shaped torch.Tensor) containing the input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''
  
  InputCheck( x, W, Print )

  MaxLag: int = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y: tor.Tensor = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state
  
  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = (2*x[k] - 0.4*tor.abs( x[k-1]**3 ) + x[k-2]*x[k-1]**2 + 0.5*tor.tan( x[k-3] ) ) / ( 0.5*tor.cos( y[k-1] ) + 2*y[k-2]**2) # those should never be zero
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = \n(2x[k] -0.4|x[k-1]^3| + x[k-2]*x[k-1]^2 + 0.5sqrt(|x[k-3]|) /\n(0.5cos(y[k-1]) + 2y[k-2]^2)")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


# ################################################################################## 3 Input MISO ##################################################################################
def ThreeInputMISO( x1: tor.Tensor, x2: tor.Tensor, x3: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = False
                  ) -> Tuple[tor.Tensor, tor.Tensor, tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  '''y[k] = 0.2x1[k] + 0.3x2[k]^3 + 0.7|x3[k]| + 0.5x2[k-3]*x1[k-2] - 0.3y[k-1] * x2[k-2]^2 - 0.4|x3[k-1] * y[k-2]^2| - 0.4x1[k-1] * x2[k-1]^2
  
  ### Inputs:
  - `x1`: ((p,)-shaped torch.Tensor) containing the first input sequence
  - `x2`: ((p,)-shaped torch.Tensor) containing the second input sequence
  - `x3`: ((p,)-shaped torch.Tensor) containing the third input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x1`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut first input sequence
  - `x2`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut second input sequence
  - `x3`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut third input sequence
  - `y`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x1, W, Print )
  InputCheck( x2, W, Print )
  InputCheck( x3, W, Print )

  if ( ( x1.shape != x2.shape ) or ( x2.shape != x3.shape ) or ( x1.shape != x3.shape ) ): raise ValueError( 'x1, x2, x3 must have the same shape' )

  MaxLag: int = 3
  x1 -= tor.mean( x1 ) # necessary
  x2 -= tor.mean( x2 ) # necessary
  x3 -= tor.mean( x3 ) # necessary
  y: tor.Tensor = tor.zeros( len( x1 ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x1 ) ): # maximum lag is 3
    y[k] = 0.2*x1[k] + 0.3*x2[k]**3 + 0.7*tor.abs( x3[k] ) + 0.5*x2[k-3]*x1[k-2] - 0.3*y[k-1]*x2[k-2]**2 - 0.8*tor.abs( x3[k-1]*y[k-2] ) - 0.7*x1[k-1]*x2[k-1]**2 # AOrLSR benchmark version
    if ( W is not None ): y[k] += W[k] # Additive Noise

  if ( Print ): print( "System: y[k] = 0.2 x1[k] + 0.3 x2[k]^3 + 0.7|x3[k]| + 0.5 x2[k-3] x1[k-2] - 0.3 y[k-1] * x2[k-2]^2 - 0.8 | x3[k-1] * y[k-2] | - 0.7 x1[k-1] * x2[k-1]^2" )
  
  if ( W is not None ): W = W[ MaxLag : ]
  return ( x1[ MaxLag : ], x2[ MaxLag : ], x3[ MaxLag : ], y[ MaxLag : ], W )



# ################################################################################## 3 Input MIMO ##################################################################################
def ThreeInputMIMO( x1: tor.Tensor, x2: tor.Tensor, x3: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = False
                  ) -> Tuple[tor.Tensor, tor.Tensor, tor.Tensor, tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  ''' y[k] = 0.2x1[k] + 0.3x2[k]^3 + 0.7|x3[k]| + 0.5x2[k-3]*x1[k-2] - 0.3y[k-1] * x2[k-2]^2 - 0.4|x3[k-1] * y[k-2]^2| - 0.4x1[k-1] * x2[k-1]^2
  
  ### Inputs:
  - `x1`: ((p,)-shaped torch.Tensor) containing the first input sequence
  - `x2`: ((p,)-shaped torch.Tensor) containing the second input sequence
  - `x3`: ((p,)-shaped torch.Tensor) containing the third input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x1`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut first input sequence
  - `x2`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut second input sequence
  - `x3`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut third input sequence
  - `y1`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response first channel
  - `y2`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response second channel
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''

  InputCheck( x1, W, Print )
  InputCheck( x2, W, Print )
  InputCheck( x3, W, Print )

  if ( ( x1.shape != x2.shape ) or ( x2.shape != x3.shape ) or ( x1.shape != x3.shape ) ): raise ValueError( 'x1, x2, x3 must have the same shape' )

  MaxLag: int = 3
  x1 -= tor.mean( x1 ) # necessary
  x2 -= tor.mean( x2 ) # necessary
  x3 -= tor.mean( x3 ) # necessary

  y1: tor.Tensor  = tor.zeros( len( x1 ) ) # system output, 0 to emulate the initialization state
  y2: tor.Tensor  = tor.zeros( len( x1 ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x1 ) ): # maximum lag is 3
    y1[k] = 0.2*x1[k]   + 0.3*x2[k]**3 + 0.7*tor.abs( x3[k] ) + 0.5*x2[k-3]*x1[k-2] - 0.3*y2[k-1]*x2[k-2]**2 - 0.8*tor.abs( x3[k-1]*y1[k-2] ) - 0.7*x1[k-1]*x2[k-1]**2
    y2[k] = 0.3*x1[k-1] + 0.5*x3[k]**3 + 0.7*tor.abs( y1[k-1] ) + 0.6*y1[k-3]*x1[k-2] - 0.4*y1[k-1]*x3[k-2]**2 - 0.9*tor.abs( x3[k-1]*y2[k-2] ) - 0.7*x3[k-1]*x2[k-1]**2
    
    if ( W is not None ): y1[k] += W[k]; y2[k] += W[k] # Additive Noise

  if ( Print ): print( "System:\n"
                      + "y1[k] = 0.2 x1[k]   + 0.3*x2[k]^3 + 0.7*|x3[k]|   + 0.5*x2[k-3]*x1[k-2] - 0.3*y2[k-1]*x2[k-2]^2 - 0.8*|x3[k-1]*y1[k-2]| - 0.7*x1[k-1]*x2[k-1]^2\n"
                      + "y2[k] = 0.3*x1[k-1] + 0.5*x3[k]^3 + 0.7*|y1[k-1]| + 0.6*y1[k-3]*x1[k-2] - 0.4*y1[k-1]*x3[k-2]^2 - 0.9*|x3[k-1]*y2[k-2]| - 0.7*x3[k-1]*x2[k-1]^2\n\n"
                     )
  
  if ( W is not None ): W = W[ MaxLag : ]
  return ( x1[ MaxLag : ], x2[ MaxLag : ], x3[ MaxLag : ], y1[ MaxLag : ], y2[ MaxLag : ], W )



# ############################################################################### Binary MISO System ###############################################################################
def Binary_MISO_System( x1: tor.Tensor, x2: tor.Tensor, x3: tor.Tensor, x4: tor.Tensor, W: Optional[tor.Tensor] = None, Print: bool = False
                      ) -> Tuple[tor.Tensor, tor.Tensor, tor.Tensor, tor.Tensor, tor.Tensor, Optional[tor.Tensor]]:
  
  '''y[k] = ( !x1[k] && x2[k] ) - ( x3[k] ^ x4[k] ) + ( x1[k] || x3[k-1] ) "
                                "+ ( x2[k] ^ x4[k-2] ) - ( !x1[k-2] && x2[k] ) - !x3[k-2] + !x2[k-1] + x4[k-1]
  
  ### Inputs:
  - `x1`: ((p,)-shaped torch.Tensor) containing the first input sequence
  - `x2`: ((p,)-shaped torch.Tensor) containing the second input sequence
  - `x3`: ((p,)-shaped torch.Tensor) containing the third input sequence
  - `W`: ((p,)-shaped torch.Tensor) containing the additive noise
  - `Print`: (bool) containing whether to print the System equation

  ### Outputs:
  - `x1`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut first input sequence
  - `x2`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut second input sequence
  - `x3`: ((p-MaxLags,)-shaped torch.Tensor) containing the centered and cut third input sequence
  - `y1`: ((p-MaxLags,)-shaped torch.Tensor) containing the System response
  - `W`: ((p-MaxLags,)-shaped torch.Tensor) containing the cut additive noise sequence
  '''
  
  for x in [ x1, x2, x3, x4 ]:
    if ( not tor.all( tor.logical_or( x == 0, x == 1 ) ).item() ): raise ValueError( "x must be a binary torch.Tensor" )
    if ( len( x ) != len( x1 ) ): raise ValueError( "All inputs must have the same length" )

  MaxLag: int = 2
  y: tor.Tensor = tor.zeros( len( x1 ), dtype = tor.int32 ) # system output, 0 to emulate the initialization state

  XOR: Callable[[tor.Tensor, tor.Tensor], tor.Tensor] = lambda x,y: tor.logical_xor( x, y ).int()
  AND: Callable[[tor.Tensor, tor.Tensor], tor.Tensor] = lambda x,y: tor.logical_and( x, y ).int()
  OR: Callable[[tor.Tensor, tor.Tensor], tor.Tensor] = lambda x,y: tor.logical_or( x, y ).int()
  NOT: Callable[[tor.Tensor], tor.Tensor] = lambda x: tor.logical_not( x ).int()
  
  for k in range( MaxLag, len( x1 ) ): # maximum lag is 1
    y[k] = ( AND( NOT( x1[k] ), x2[k] ) - XOR( x3[k], x4[k] ) + OR( x1[k], x3[k-1] )
           + XOR( x2[k], x4[k-2] ) - AND( NOT( x1[k-2] ), x2[k] ) - NOT( x3[k-2] ) + NOT( x2[k-1] ) + x4[k-1]
           )
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print( "System: y[k] = ( !x1[k] && x2[k] ) - ( x3[k] ^ x4[k] ) + ( x1[k] || x3[k-1] ) "
                                "+ ( x2[k] ^ x4[k-2] ) - ( !x1[k-2] && x2[k] ) - !x3[k-2] + !x2[k-1] + x4[k-1] \n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x1[ MaxLag : ], x2[ MaxLag : ], x3[ MaxLag : ], x4[ MaxLag : ], y[ MaxLag : ], W )