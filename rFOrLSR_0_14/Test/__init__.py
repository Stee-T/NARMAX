# FOrLSR.Test file
import torch as tor

def InputCheck( x, W, Print ):
  if ( not isinstance( x, tor.Tensor ) ): raise ValueError( "x must be a torch.Tensor" )
  if ( x.dim() != 1 ): raise ValueError( "x must be a 1D torch.Tensor" )

  if ( W is not None ): 
    if ( ( not isinstance( W, tor.Tensor ) ) or ( W.dim() != x.dim() ) ): raise ValueError( "W must be None or a torch.Tensor of the same dimension as x" )

  if ( not isinstance( Print, bool ) ): raise ValueError( "Print must be a bool" )


def ARX( x, W = None, Print = True ):
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

  MaxLag = 4
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 4
    y[k] = 0.2*x[k] + 0.7*x[k-1] + 0.6*x[k-2] +0.4*x[k-3] -0.5*x[k-4] -0.5*y[k-1] -0.7*y[k-2] -0.3*y[k-3] + 0.3*y[k-4]
    if ( W is not None ): y[k] += W[k] # Additive Noise 
  
  if ( Print ): print("System: y[k] = 0.2*x[k] + 0.7*x[k-1] + 0.6*x[k-2] +0.4*x[k-3] -0.5*x[k-4] -0.5*y[k-1] -0.7*y[k-2] -0.3*y[k-3] + 0.3*y[k-4] \n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


def TermCombinations( x, W = None, Print = True ):
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

  MaxLag = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = -0.2*x[k] - 0.4*x[k-1]**3 + 0.3*x[k-2]*x[k-1]**2 + 0.5*x[k-3]*x[k-2] + 0.5*y[k-1] + 0.2*y[k-2]**2 -0.1*y[k-3]**3
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = -0.2x[k] -0.4x[k-1]^3 + 0.3x[k-2]*x[k-1]^2 + 0.5x[k-3]*x[k-2] + 0.5y[k-1] + 0.2y[k-2]^2 - 0.1y[k-3]^3\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


def iFOrLSR( x, W = None, Print = True ): # iFOrLSR paper example
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

  MaxLag = 2
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 2
    y[k] = 0.2*y[k-1]**3 + 0.7*y[k-1]*x[k-1] + 0.6*x[k-2]**2 -0.5*y[k-2] -0.7*y[k-2]*x[k-2]**2
    if ( W is not None ): y[k] += W[k] # Additive Noise 

  if ( Print ): print("System: y[k] = 0.2y[k-1]^3 + 0.7y[k-1]*x[k-1] +0.6x[k-2]^2 -0.5y[k-2] -0.7y[k-2]*x[k-2]^2\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


def NonLinearities( x, W = None, Print = True ):
  '''y[k] = 0.2x[k] + 0.3x[k-1]^3 + 0.7|x[k-2]*x[k-1]^2| + 0.5exp(x[k-3]*x[k-2]) - 0.5cos(y[k-1]*x[k-2]) + 0.2|x[k-1]*y[k-2]^2| - 0.1y[k-3]^3
  
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

  MaxLag = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = 0.2*x[k] + 0.3*x[k-1]**3 + 0.7*tor.abs( x[k-2]*x[k-1]**2) + 0.5*tor.exp( x[k-3]*x[k-2] ) - 0.5*tor.cos( y[k-1] * x[k-2] ) - 0.4*tor.abs( x[k-1] * y[k-2]**2 ) - 0.4*y[k-3]**3 # AOrLSR benchmark version
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = 0.2x[k] + 0.3x[k-1]^3 + 0.7|x[k-2]*x[k-1]^2| + 0.5exp(x[k-3]*x[k-2]) - 0.5cos(y[k-1]*x[k-2]) + 0.2|x[k-1]*y[k-2]^2| - 0.1y[k-3]^3\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )
  
 
def NotInDictionaty_Test( x, W = None, Print = True ): # Model with terms not in the dictionary
  '''y[k] = 0.5cos(0.2x[k] + 0.3x[k-1])^3 + 0.2exp(x[k-2]*x[k-1])**2 + 0.5|tanh(x[k-3]*x[k-2])| + 0.1|y[k-3]|^3
    
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

  MaxLag = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state
  
  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = 0.5*tor.cos(0.2*x[k] + 0.3*x[k-1])**3 + 0.2*tor.tan( x[k-2]*x[k-1]) + 0.5*tor.abs(tor.tanh( x[k-3]*x[k-2])) + 0.1*tor.abs(y[k-3]-0.1)**3
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = 0.5cos(0.2x[k] + 0.3x[k-1])^3 + 0.2exp(x[k-2]*x[k-1])**2 + 0.5|tanh(x[k-3]*x[k-2])| + 0.1|y[k-3]|^3\n")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


def RatNonLinSystem( x, W = None, Print = True ): # Rational system used in the AOrLSR paper
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

  MaxLag = 2
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state

  for k in range( MaxLag, len( x ) ): # maximum lag is 2
    # y[k] = ( 0.6*tor.abs( x[k] ) - 0.35*x[k]**3 - 0.2*x[k-1]**2 + 0.1*tor.abs( y[k-1] )  ) / ( 1 - 0.4*tor.abs( x[k] ) + 0.3*tor.abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] ) # OLD Paper
    y[k] = ( 0.6*tor.abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*tor.abs( y[k-1] )  ) / ( 1 - 0.4*tor.abs( x[k] ) + 0.3*tor.abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] ) # With mixed terms
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  # if ( Print ): print("y[k] = ( 0.6*tor.abs( x[k] ) - 0.35*x[k]**3 - 0.2*x[k-1]**2 + 0.1*tor.abs( y[k-1] )  ) / ( 1 - 0.4*tor.abs( x[k] ) + 0.3*tor.abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] )") # paper version
  if ( Print ): print("System: y[k] = ( 0.6*abs( x[k] ) - 0.35*x[k]**3 - 0.3*x[k-1]*y[k-2] + 0.1*abs( y[k-1] )  ) / ( 1 - 0.4*abs( x[k] ) + 0.3*abs( x[k-1]*x[k] ) - 0.2*x[k-1]**3 + 0.3*y[k-1]*x[k-2] )")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )


def RationalNARMAX( x, W = None, Print = True ):
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

  MaxLag = 3
  x -= tor.mean( x ) # necessary else all multiplications are ( x/y[..] + mean)*( x/y[..] + mean), which doesn't correspond to what's desired or in the dict
  y = tor.zeros( len( x ) ) # system output, 0 to emulate the initialization state
  
  for k in range( MaxLag, len( x ) ): # maximum lag is 3
    y[k] = (2*x[k] - 0.4*tor.abs( x[k-1]**3) + x[k-2]*x[k-1]**2 + 0.5*tor.tan( x[k-3])) / (0.5*tor.cos(y[k-1]) + 2*y[k-2]**2) # those should never be zero
    if ( W is not None ): y[k] += W[k] # Additive Noise
  
  if ( Print ): print("System: y[k] = \n(2x[k] -0.4|x[k-1]^3| + x[k-2]*x[k-1]^2 + 0.5sqrt(|x[k-3]|) /\n(0.5cos(y[k-1]) + 2y[k-2]^2)")
  if ( W is not None ): W = W[ MaxLag : ]
  
  return ( x[ MaxLag : ], y[ MaxLag : ], W )