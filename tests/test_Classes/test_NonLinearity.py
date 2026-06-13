# tests/test_Classes/test_Nonlinearity.py
from typing import Callable
import pytest
import warnings

import torch as tor
from NARMAX.Classes.NonLinearity import NonLinearity


# ------------------------------------------------------------------------------
# Fixtures for common valid and invalid functions
# ------------------------------------------------------------------------------

@pytest.fixture
def valid_f() -> Callable:
  '''Elementwise tanh function.'''
  return lambda x: tor.tanh( x )


@pytest.fixture
def valid_fPrime() -> Callable:
  '''Derivative of tanh: 1 - tanh( x )^2.'''
  return lambda x: 1 - tor.tanh( x )**2


@pytest.fixture
def valid_fSecond() -> Callable:
  '''Second derivative of tanh: -2*tanh( x )*( 1 - tanh( x )^2 ).'''
  return lambda x: -2 * tor.tanh( x ) * ( 1 - tor.tanh( x )**2 )


@pytest.fixture
def shape_mismatch_f() -> Callable:
  '''Function that returns a different shape.'''
  return lambda x: tor.sum( x ).reshape( 1 )


@pytest.fixture
def non_tensor_f() -> Callable:
  '''Function that returns a list instead of a tensor.'''
  return lambda x: x.tolist()


@pytest.fixture
def nan_producing_f() -> Callable:
  '''Function that produces NaNs for some inputs.'''
  return lambda x: tor.log( x ) # log( negative ) -> NaN


@pytest.fixture
def time_dependent_f() -> Callable:
  '''Function that is not elementwise ( depends on sequence position ).'''
  return lambda x: x + tor.arange( len( x ), dtype = x.dtype, device = x.device )


# ------------------------------------------------------------------------------
# Valid Initialization Tests
# ------------------------------------------------------------------------------

def test_init_minimal_valid( valid_f ) -> None:
  '''Test initialization with only the required function.'''
  nl = NonLinearity( "tanh", valid_f )
  assert nl.Name == "tanh"
  assert isinstance( nl.Name, str )
  assert nl.f is valid_f
  assert isinstance( nl.f, Callable )
  assert nl.fPrime is None
  assert nl.fSecond is None
  assert nl.ToMorph is False
  assert nl.get_f() is valid_f
  assert nl.get_Name() == "tanh"
  assert nl.to_Morph() is False
  assert nl.get_fPrime() is None
  assert nl.get_fSecond() is None
  assert str( nl ) == "tanh"
  r = repr( nl )
  assert "Name=tanh" in r
  assert "Not Morphable" in r


def test_init_with_derivatives_and_tomorph_true( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''Test initialization with all derivatives and ToMorph=True.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = True )
  assert nl.Name == "tanh"
  assert isinstance( nl.Name, str )
  assert nl.f is valid_f
  assert isinstance( nl.f, Callable )
  assert nl.fPrime is valid_fPrime
  assert isinstance( nl.fPrime, Callable )
  assert nl.fSecond is valid_fSecond
  assert isinstance( nl.fSecond, Callable )
  assert nl.ToMorph is True
  assert isinstance( nl.ToMorph, bool )
  assert str( nl ) == "tanh"
  r = repr( nl )
  assert "Name=tanh" in r
  assert "Morphable" in r


def test_init_with_derivatives_and_tomorph_false( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''Test initialization with derivatives but ToMorph=False.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = False )
  assert nl.ToMorph is False
  assert nl.fPrime is valid_fPrime
  assert nl.fSecond is valid_fSecond
  assert str( nl ) == "tanh"
  assert "Not Morphable" in repr( nl )


def test_init_with_derivatives_tomorph_none_warns( valid_f, valid_fPrime, valid_fSecond, capsys ) -> None:
  '''Test that passing derivatives without ToMorph prints a warning.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = None )
  captured = capsys.readouterr()
  assert "WARNING" in captured.out
  assert nl.ToMorph is False
  assert nl.fPrime is valid_fPrime
  assert nl.fSecond is valid_fSecond
  assert nl.Name == "tanh"
  assert captured.err == ""


def test_init_tomorph_none_without_derivatives( valid_f ) -> None:
  '''ToMorph=None without derivatives should not warn.'''
  nl = NonLinearity( "tanh", valid_f, ToMorph = None )
  assert nl.ToMorph is False


def test_init_tomorph_false_without_derivatives( valid_f ) -> None:
  '''Explicit ToMorph=False without derivatives is legal.'''
  nl = NonLinearity( "tanh", valid_f, ToMorph = False )
  assert nl.ToMorph is False


# ------------------------------------------------------------------------------
# Name Validation Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize( "invalid_name", [ "", None, 123, [] ] )
def test_invalid_name_raises_valueerror( invalid_name, valid_f ) -> None:
  '''Name must be a non-empty string.'''
  with pytest.raises( ValueError, match = "Name must be a non-empty string" ):
    NonLinearity( invalid_name, valid_f )


# ------------------------------------------------------------------------------
# Callable Validation Tests
# ------------------------------------------------------------------------------

def test_non_callable_f_raises_valueerror() -> None:
  '''f must be callable.'''
  with pytest.raises( ValueError, match = "f from test function pointer must be a callable object" ):
    NonLinearity( "test", "not_a_function" )


def test_non_callable_fPrime_raises_valueerror( valid_f ) -> None:
  '''fPrime must be callable if provided.'''
  with pytest.raises( ValueError, match = "fPrime from test function pointer must be a callable object" ):
    NonLinearity( "test", valid_f, fPrime = "not_callable" )


def test_non_callable_fSecond_raises_valueerror( valid_f, valid_fPrime ) -> None:
  '''fSecond must be callable if provided.'''
  with pytest.raises( ValueError, match = "fSecond from test function pointer must be a callable object" ):
    NonLinearity( "test", valid_f, valid_fPrime, fSecond = 123 )


# ------------------------------------------------------------------------------
# Shape and Tensor Type Validation Tests
# ------------------------------------------------------------------------------

def test_f_shape_mismatch_raises_assertion( shape_mismatch_f ) -> None:
  '''f must return a tensor of the same shape as input.'''
  with pytest.raises( AssertionError, match = "f from test must return torch tensors of the same size" ):
    NonLinearity( "test", shape_mismatch_f )


def test_f_non_tensor_raises_assertion( non_tensor_f ) -> None:
  '''f must return a torch.Tensor.'''
  with pytest.raises( AssertionError, match = "f from test must return torch tensors of the same size" ):
    NonLinearity( "test", non_tensor_f )


def test_fPrime_shape_mismatch_raises_assertion( valid_f, shape_mismatch_f ) -> None:
  '''fPrime must return same shape tensor.'''
  with pytest.raises( AssertionError, match = "fPrime from test must return torch tensors of the same size" ):
    NonLinearity( "test", valid_f, fPrime = shape_mismatch_f )


def test_fPrime_non_tensor_raises_assertion( valid_f, non_tensor_f ) -> None:
  '''fPrime must return a torch.Tensor.'''
  with pytest.raises( AssertionError, match = "fPrime from test must return torch tensors of the same size" ):
    NonLinearity( "test", valid_f, fPrime = non_tensor_f )


def test_fSecond_shape_mismatch_raises_assertion( valid_f, valid_fPrime, shape_mismatch_f ) -> None:
  '''fSecond must return same shape tensor.'''
  with pytest.raises( AssertionError, match = "fSecond from test must return torch tensors of the same size" ):
    NonLinearity( "test", valid_f, valid_fPrime, fSecond = shape_mismatch_f )


def test_fSecond_non_tensor_raises_assertion( valid_f, valid_fPrime, non_tensor_f ) -> None:
  '''fSecond must return a torch.Tensor.'''
  with pytest.raises( AssertionError, match = "fSecond from test must return torch tensors of the same size" ):
    NonLinearity( "test", valid_f, valid_fPrime, fSecond = non_tensor_f )


# ------------------------------------------------------------------------------
# NaN Detection ( Warnings )
# ------------------------------------------------------------------------------

# NaN Detection ( Warnings ) – now expects warnings, not exceptions
def test_f_with_nans_raises_warning( nan_producing_f ) -> None:
  '''Functions that produce NaNs should raise a warning.'''
  with pytest.warns( Warning, match = "f from test generates NaN values" ):
    NonLinearity( "test", nan_producing_f )


def test_fPrime_with_nans_raises_warning( valid_f, nan_producing_f ) -> None:
  '''fPrime that produces NaNs should raise a warning.'''
  with pytest.warns( Warning, match = "fPrime from test generates NaN values" ):
    NonLinearity( "test", valid_f, fPrime = nan_producing_f )


def test_fSecond_with_nans_raises_warning( valid_f, valid_fPrime, nan_producing_f ) -> None:
  '''fSecond that produces NaNs should raise a warning.'''
  with pytest.warns( Warning, match = "fSecond from test generates NaN values" ):
    NonLinearity( "test", valid_f, valid_fPrime, fSecond = nan_producing_f )


def test_no_nan_warning_for_clean_functions( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''Clean functions should not produce NaN warnings.'''
  with warnings.catch_warnings( record = True ) as record:
    NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = True )
  warnings_raised = [ w for w in record if issubclass( w.category, Warning ) ]
  assert len( warnings_raised ) == 0, f"Unexpected warnings: { warnings_raised }"


# ------------------------------------------------------------------------------
# Time Dependency ( Elementwise ) Validation
# ------------------------------------------------------------------------------

def test_time_dependent_f_raises_assertion( time_dependent_f ) -> None:
  '''Non-elementwise functions should raise AssertionError.'''
  with pytest.raises( AssertionError, match = "f from test should be elementwise" ):
    NonLinearity( "test", time_dependent_f )


def test_elementwise_f_passes( valid_f ) -> None:
  '''Elementwise functions should not raise time dependency error.'''
  # Should not raise
  NonLinearity( "test", valid_f )


# ------------------------------------------------------------------------------
# ToMorph Logic Validation
# ------------------------------------------------------------------------------

def test_tomorph_true_without_derivatives_raises_valueerror( valid_f ) -> None:
  '''ToMorph=True requires fPrime and fSecond.'''
  with pytest.raises( ValueError, match = "If fPrime or fSecond is None, then ToMorph must be False" ):
    NonLinearity( "test", valid_f, ToMorph = True )


def test_tomorph_true_with_only_fprime_raises_valueerror( valid_f, valid_fPrime ) -> None:
  '''ToMorph=True requires both fPrime and fSecond.'''
  with pytest.raises( ValueError, match = "If fPrime or fSecond is None, then ToMorph must be False" ):
    NonLinearity( "test", valid_f, valid_fPrime, ToMorph = True )


def test_tomorph_true_with_only_fsecond_raises_valueerror( valid_f, valid_fSecond ) -> None:
  '''ToMorph=True requires both fPrime and fSecond.'''
  with pytest.raises( ValueError, match = "If fPrime or fSecond is None, then ToMorph must be False" ):
    NonLinearity( "test", valid_f, fSecond = valid_fSecond, ToMorph = True )


# ------------------------------------------------------------------------------
# Getters Tests
# ------------------------------------------------------------------------------

def test_getters( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''Test that getters return the correct values.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = True )
  assert nl.get_f() is valid_f
  assert isinstance( nl.get_f(), Callable )
  assert nl.get_fPrime() is valid_fPrime
  assert isinstance( nl.get_fPrime(), Callable )
  assert nl.get_fSecond() is valid_fSecond
  assert isinstance( nl.get_fSecond(), Callable )
  assert nl.get_Name() == "tanh"
  assert isinstance( nl.get_Name(), str )
  assert nl.to_Morph() is True
  assert isinstance( nl.to_Morph(), bool )


def test_getters_return_none_for_missing_derivatives( valid_f ) -> None:
  '''Getters should return None for fPrime/fSecond when not provided.'''
  nl = NonLinearity( "tanh", valid_f )
  assert nl.get_fPrime() is None
  assert nl.get_fSecond() is None
  assert nl.to_Morph() is False


# ------------------------------------------------------------------------------
# Immutability ( No Setters )
# ------------------------------------------------------------------------------

def test_attributes_are_assignable_but_getters_no_setters( valid_f ) -> None:
  '''The class has no setter methods, but attributes can be reassigned directly ( Python default ).'''
  nl = NonLinearity( "tanh", valid_f )
  nl.f = lambda x: x * 2
  assert nl.get_f()( tor.tensor( 3.0 ) ) == 6.0
  for setter in [ "set_f", "set_Name", "set_fPrime", "set_fSecond", "set_ToMorph", "set_to_Morph" ]:
    assert not hasattr( nl, setter ), f"Unexpected setter method: { setter }"


# ------------------------------------------------------------------------------
# Edge Cases: Different dtypes and devices
# ------------------------------------------------------------------------------

def test_validation_works_with_different_dtypes( valid_f ) -> None:
  '''Validation should succeed even if input tensor has a different dtype ( implicitly cast ).'''
  # The internal test uses torch.randn( 100 ) -> float32.
  # We just ensure no error is raised for valid function.
  NonLinearity( "tanh", valid_f )


def test_f_handles_integer_input( valid_f ) -> None:
  '''The function should handle integer tensor inputs without errors.'''
  nl = NonLinearity( "tanh", valid_f )
  int_tensor = tor.tensor( [ 1, 2, 3 ] )
  result = nl.f( int_tensor )
  assert result.dtype in ( tor.float32, tor.float64 )
  assert result.shape == int_tensor.shape
  assert not tor.isnan( result ).any()


def test_f_handles_negative_input( valid_f ) -> None:
  '''The function should handle negative input values correctly.'''
  nl = NonLinearity( "tanh", valid_f )
  neg_tensor = tor.tensor( [ -5.0, -1.0, -0.5, 0.0 ] )
  result = nl.f( neg_tensor )
  expected = tor.tanh( neg_tensor )
  assert result.dtype == expected.dtype
  assert tor.allclose( result, expected )
  assert ( result[ neg_tensor < 0 ] < 0 ).all()
  assert result[ -1 ].item() == 0.0


# ------------------------------------------------------------------------------
# __repr__ and __str__ Tests
# ------------------------------------------------------------------------------

def test_repr_includes_name_and_derivatives_and_morph_status( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''__repr__ should contain Name, derivative info, and morph status.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = True )
  r = repr( nl )
  assert r == "NonLinearity(Name=tanh, derivatives=[1st, 2nd], Morphable)"
  assert "Name=tanh" in r
  assert "1st" in r
  assert "2nd" in r
  assert "Morphable" in r


def test_repr_without_derivatives( valid_f ) -> None:
  '''__repr__ should indicate no derivatives.'''
  nl = NonLinearity( "relu", valid_f )
  r = repr( nl )
  assert r == "NonLinearity(Name=relu, derivatives=[None], Not Morphable)"
  assert "Name=relu" in r
  assert "None" in r
  assert "Not Morphable" in r


def test_repr_with_only_fprime( valid_f, valid_fPrime ) -> None:
  '''__repr__ with only fPrime should list only 1st derivative.'''
  nl = NonLinearity( "test", valid_f, fPrime = valid_fPrime )
  r = repr( nl )
  assert r == "NonLinearity(Name=test, derivatives=[1st], Not Morphable)"
  assert "1st" in r
  assert "2nd" not in r
  assert "Not Morphable" in r


def test_str_returns_name( valid_f ) -> None:
  '''__str__ should return just the name.'''
  nl = NonLinearity( "tanh", valid_f )
  assert str( nl ) == "tanh"
  assert isinstance( str( nl ), str )


def test_str_with_derivatives( valid_f, valid_fPrime, valid_fSecond ) -> None:
  '''__str__ should still only return the name, regardless of derivatives.'''
  nl = NonLinearity( "tanh", valid_f, valid_fPrime, valid_fSecond, ToMorph = True )
  assert str( nl ) == "tanh"
  assert str( nl ) != repr( nl )


def test_str_returns_name_for_custom_name( valid_f ) -> None:
  '''__str__ should return the custom name.'''
  nl = NonLinearity( "custom_name", valid_f )
  assert str( nl ) == "custom_name"


def test_custom_elementwise_function() -> None:
  '''Test a custom elementwise function that meets all criteria.'''
  def custom_f( x ): return x**2

  def custom_fPrime( x ): return 2 * x

  def custom_fSecond( x ): return 2 * tor.ones_like( x )

  nl = NonLinearity( "square", custom_f, custom_fPrime, custom_fSecond, ToMorph = True )
  x = tor.tensor( [ 1.0, 2.0, 3.0 ] )
  assert tor.allclose( nl.f( x ), tor.tensor( [ 1.0, 4.0, 9.0 ] ) )
  assert tor.allclose( nl.fPrime( x ), tor.tensor( [ 2.0, 4.0, 6.0 ] ) )
  assert tor.allclose( nl.fSecond( x ), tor.tensor( [ 2.0, 2.0, 2.0 ] ) )
  assert nl.get_Name() == "square"
  assert nl.to_Morph() is True
  assert str( nl ) == "square"
  r = repr( nl )
  assert "Name=square" in r
  assert "Morphable" in r
  assert isinstance( nl.get_f(), Callable )
  assert isinstance( nl.get_fPrime(), Callable )
  assert isinstance( nl.get_fSecond(), Callable )


def test_nan_warning_still_creates_object( nan_producing_f ) -> None:
  '''NaN warning should not prevent object creation.'''
  with pytest.warns( Warning, match = "f from test generates NaN values" ):
    nl = NonLinearity( "test", nan_producing_f )
  assert nl.Name == "test"
  assert nl.f is nan_producing_f
  assert nl.fPrime is None
  assert nl.fSecond is None
  assert nl.ToMorph is False
