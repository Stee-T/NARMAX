import torch as tor
from typing import Optional, Callable

class NonLinearity:
  ###################################################################################### Init ######################################################################################
  def __init__( self, Name: str,
                f: Callable[ [ tor.Tensor ], tor.Tensor], # The function itself
                fPrime: Optional[ Callable[ [ tor.Tensor ], tor.Tensor] ] = None, fSecond: Optional[ Callable[ [ tor.Tensor ], tor.Tensor] ] = None, # First and second derivatives
                ToMorph: Optional[ bool ] = None ) -> None:

    if ( ( not Name ) or ( not isinstance( Name, str ) ) ): raise ValueError( "Name must be a non-empty string." )


    # Check if function pointers are callable objects
    Error = f" from { Name } function pointer must be a callable object"
    if ( not callable( f ) ):         raise ValueError( "f" + Error )
    
    if ( fPrime is not None ):
      if ( not callable( fPrime ) ):  raise ValueError( "fPrime" + Error )
    
    if ( fSecond is not None ):
      if ( not callable( fSecond ) ): raise ValueError( "fSecond" + Error )


    # Check if functions returns torch tensors of the same size
    input_tensor = tor.randn( 100 )
    Error =  f" from { Name } must return torch tensors of the same size as their input."
    if ( ( not isinstance( f( input_tensor ), tor.Tensor ) ) or ( f( input_tensor ).shape != input_tensor.shape) ):
      raise AssertionError( "f" + Error )
    
    if ( fPrime is not None ):
      Output = fPrime( input_tensor )
      if ( ( not isinstance( Output, tor.Tensor ) ) or ( Output.shape != input_tensor.shape) ):
        raise AssertionError( "fPrime" + Error )
      
    if ( fSecond is not None ):
      Output = fSecond( input_tensor )
      if ( ( not isinstance( Output, tor.Tensor ) ) or ( Output.shape != input_tensor.shape) ):
        raise AssertionError( "fSecond" + Error )


    # Check for NaNs for a small range
    Error = f" from { Name } generates NaN values for an input around 0 with variance 1. This could become problematic for the morphing"
    if ( tor.isnan( f( input_tensor ) ).any() ):          raise Warning( "f" + Error )

    if ( fPrime is not None ):
      if ( tor.isnan( fPrime( input_tensor ) ).any() ):   raise Warning( "fPrime" + Error )
    
    if ( fSecond is not None ):
      if ( tor.isnan( fSecond( input_tensor ) ).any() ):  raise Warning( "fSecond" + Error )


    # Check for time dependencies
    time_tensor = f( tor.ones( 500 ) )
    if ( not tor.allclose( time_tensor, time_tensor[0] * tor.ones( 500 ) ) ) : # All values should be exactly the same if there is no dependency
      raise AssertionError( f"f from { Name } should be elementwise, thus not have time dependencies." )


    self.Name: str = Name
    self.f: Callable[ [ tor.Tensor ], tor.Tensor ] = f
    self.fPrime: Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ] = fPrime
    self.fSecond: Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ] = fSecond

    if ( ( fPrime is None ) or ( fSecond is None ) ):
      if ( ToMorph is not None ):
        if ( ToMorph is True ): raise ValueError( "If fPrime or fSecond is None, then ToMorph must be False, as wee need those for the morphing" )
        else:                       self.ToMorph: bool = False # User precised that no morphing is necessary, useless but legal
      else: self.ToMorph: bool = False

    else: # fPrime and fSecond are both passed and valid
      if ( ToMorph is None ):
        print("\n\n WARNING: fPrime and fSecond were passed but no information if morphing is desired was given. Defaults to false\n\n")
        self.ToMorph: bool = False
      else: self.ToMorph: bool = ToMorph # Allows the user to not morph that non-linearity
 
  ############################################################### Getters for function pointers and other attributes ###############################################################
  # No setters defined making th objecst implicitly const
  def get_f( self ):        return ( self.f )
  def get_fPrime( self ):   return ( self.fPrime )
  def get_fSecond( self ):  return ( self.fSecond )
  def get_Name( self ):     return ( self.Name )
  def to_Morph( self ): return ( self.ToMorph )