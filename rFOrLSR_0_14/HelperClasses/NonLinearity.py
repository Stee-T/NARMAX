import torch as tor

class NonLinearity:
  def __init__( self, Name, f, fPrime = None, fSecond = None, IsMorphable = True ):

    # Check if the passed string is non-empty
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


    self.Name = Name
    self.f = f
    self.fPrime = fPrime
    self.fSecond = fSecond

    if ( ( fPrime is None ) or ( fSecond is None ) ): self.IsMorphable = False
    else:                                             self.IsMorphable = IsMorphable # Allows the user to not morph that non-linearity

  # Getters for function pointers and other attributes
  def get_f( self ):        return ( self.f )
  def get_fPrime( self ):   return ( self.fPrime )
  def get_fSecond( self ):  return ( self.fSecond )
  def get_Name( self ):     return ( self.Name )
  def is_morphable( self ): return ( self.IsMorphable )