import torch as tor
import warnings
from typing import Optional, Callable

class NonLinearity:
  '''Represents a non-linear function with its optional first and second derivatives.'''
  ###################################################################################### Init ######################################################################################
  def __init__( self, Name: str,
                f: Callable[ [ tor.Tensor ], tor.Tensor ], # The function itself
                fPrime: Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ] = None, fSecond: Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ] = None, # First and second derivatives
                ToMorph: Optional[ bool ] = None ) -> None:
    '''Initialize a NonLinearity instance.

    Args:
        Name: The name of the non-linearity.
        f: The function itself.
        fPrime: Optional first derivative.
        fSecond: Optional second derivative.
        ToMorph: Whether morphing is desired.

    Raises:
        ValueError: If Name is empty, or if function pointers are not callable,
            or if ToMorph is True but derivatives are missing.
        AssertionError: If functions do not return tensors of the same shape,
            or if f has time dependencies.
    '''

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
    Error = f" from { Name } must return torch tensors of the same size as their input."
    if ( ( not isinstance( f( input_tensor ), tor.Tensor ) ) or ( f( input_tensor ).shape != input_tensor.shape ) ):
      raise AssertionError( "f" + Error )

    if ( fPrime is not None ):
      Output = fPrime( input_tensor )
      if ( ( not isinstance( Output, tor.Tensor ) ) or ( Output.shape != input_tensor.shape ) ):
        raise AssertionError( "fPrime" + Error )

    if ( fSecond is not None ):
      Output = fSecond( input_tensor )
      if ( ( not isinstance( Output, tor.Tensor ) ) or ( Output.shape != input_tensor.shape ) ):
        raise AssertionError( "fSecond" + Error )


    # Check for NaNs for a small range
    Error = f" from { Name } generates NaN values for an input around 0 with variance 1. This could become problematic for the morphing"
    if ( tor.isnan( f( input_tensor ) ).any() ): warnings.warn( "f" + Error, Warning )

    if ( fPrime is not None ):
      if ( tor.isnan( fPrime( input_tensor ) ).any() ): warnings.warn( "fPrime" + Error, Warning )

    if ( fSecond is not None ):
      if ( tor.isnan( fSecond( input_tensor ) ).any() ): warnings.warn( "fSecond" + Error, Warning )


    # Check for time dependencies
    time_tensor = f( tor.ones( 500 ) )
    if ( not tor.allclose( time_tensor, time_tensor[ 0 ] * tor.ones( 500 ) ) ): # All values should be exactly the same if there is no dependency
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
        print( "\n\n WARNING: fPrime and fSecond were passed but no information if morphing is desired was given. Defaults to false\n\n" )
        self.ToMorph: bool = False
      else: self.ToMorph: bool = ToMorph # Allows the user to not morph that non-linearity

  def __repr__( self ) -> str:
    '''Return a string representation of the NonLinearity.'''
    derivatives = []
    if ( self.fPrime is not None ): derivatives.append( "1st" )
    if ( self.fSecond is not None ): derivatives.append( "2nd" )
    deriv_str = ", ".join( derivatives ) if derivatives else "None"
    morph_str = "Morphable" if self.ToMorph else "Not Morphable"
    return ( f"NonLinearity(Name={ self.Name }, derivatives=[{ deriv_str }], { morph_str })" )

  def __str__( self ) -> str:
    '''Return the name of the NonLinearity.'''
    return ( self.Name )

  ############################################################### Getters for function pointers and other attributes ###############################################################
  # No setters defined making the objects implicitly const
  def get_f( self ) -> Callable[ [ tor.Tensor ], tor.Tensor ]:
    '''Return the function pointer.'''
    return ( self.f )

  def get_fPrime( self ) -> Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ]:
    '''Return the first derivative function pointer.'''
    return ( self.fPrime )

  def get_fSecond( self ) -> Optional[ Callable[ [ tor.Tensor ], tor.Tensor ] ]:
    '''Return the second derivative function pointer.'''
    return ( self.fSecond )

  def get_Name( self ) -> str:
    '''Return the name of the non-linearity.'''
    return ( self.Name )

  def to_Morph( self ) -> bool:
    '''Return whether this non-linearity can be morphed.'''
    return ( self.ToMorph )
