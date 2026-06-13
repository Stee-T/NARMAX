import math
import pytest
from NARMAX.HelperFuncs import Combinations # replace with actual module

class TestCombinations:
  '''Tests for the corrected Combinations function.'''

  # --- Correctness against math.comb ---
  @pytest.mark.parametrize( "n, k, expected", [
        ( 5, 2, 10 ),
        ( 6, 4, 15 ),
        ( 10, 3, 120 ),
        ( 10, 7, 120 ),
        ( 0, 0, 1 ),
        ( 1, 0, 1 ),
        ( 1, 1, 1 ),
        ( 4, 4, 1 ),
        ( 30, 15, math.comb( 30, 15 ) ), # large value, exact
        ( 50, 25, math.comb( 50, 25 ) ),
    ] )
  def test_combinations_match_math( self, n: int, k: int, expected: int ) -> None:
    '''Verify Combinations matches math.comb for various inputs.'''
    result = Combinations( n, k )
    assert result == expected
    assert isinstance( result, int )

  # --- k > N returns 0 ---
  @pytest.mark.parametrize( "n, k", [
        ( 3, 7 ),
        ( 0, 1 ),
        ( 5, 10 ),
    ] )
  def test_k_greater_than_N_returns_zero( self, n: int, k: int ) -> None:
    '''Returns 0 when k exceeds n.'''
    result = Combinations( n, k )
    assert result == 0
    assert isinstance( result, int )

  # --- Type checks ---
  def test_float_input_raises_type_error( self ) -> None:
    '''Float arguments raise TypeError.'''
    with pytest.raises( TypeError, match = "must be integers" ):
      Combinations( 5.5, 2 )

  def test_string_input_raises_type_error( self ) -> None:
    '''String arguments raise TypeError.'''
    with pytest.raises( TypeError, match = "must be integers" ):
      Combinations( "5", 2 )

  # --- Bool inputs (bool is subclass of int, so accepted) ---
  @pytest.mark.parametrize( "n, k, expected", [
        ( True, 0, 1 ),   # C(1, 0)
        ( True, True, 1 ),# C(1, 1)
        ( False, False, 1 ),# C(0, 0)
        ( True, 5, 0 ),   # k > n
    ] )
  def test_bool_inputs_are_accepted( self, n: bool, k: bool, expected: int ) -> None:
    '''Bool arguments work because bool is a subclass of int.'''
    result = Combinations( n, k )
    assert result == expected
    assert isinstance( result, int )

  # --- Other invalid types ---
  @pytest.mark.parametrize( "n, k", [
        ( [ 1, 2 ], 2 ),
        ( { "a": 1 }, 2 ),
        ( None, 2 ),
        ( 5, [ 1, 2 ] ),
        ( 5, None ),
    ] )
  def test_invalid_types_raise_type_error( self, n, k ) -> None:
    '''Non-integer, non-bool types raise TypeError.'''
    with pytest.raises( TypeError, match = "must be integers" ):
      Combinations( n, k )

  # --- Negativity checks (bug fixed: now raises ValueError, not 0) ---
  @pytest.mark.parametrize( "n, k", [
        ( -1, 0 ), # previously returned 0
        ( -2, -1 ), # previously returned 0
        ( 5, -2 ),
        ( -5, 3 ),
    ] )
  def test_negative_input_raises_value_error( self, n: int, k: int ) -> None:
    '''Negative inputs raise ValueError.'''
    with pytest.raises( ValueError, match = "negative" ):
      Combinations( n, k )

  # --- Large inputs (stress test) ---
  def test_large_combination_no_precision_loss( self ) -> None:
    '''Large combination C(1000,500) computed exactly.'''
    # C(1000, 500) – a number with ~300 digits – must be exact
    result = Combinations( 1000, 500 )
    assert result == math.comb( 1000, 500 )
    assert isinstance( result, int )
    assert result > 0

  # --- Positive result for all valid inputs ---
  @pytest.mark.parametrize( "n, k", [
        ( 0, 0 ),
        ( 3, 1 ),
        ( 7, 3 ),
        ( 12, 5 ),
        ( 20, 10 ),
        ( 30, 2 ),
    ] )
  def test_valid_input_returns_positive_int( self, n: int, k: int ) -> None:
    '''Combinations(n, k) >= 1 for 0 <= k <= n.'''
    result = Combinations( n, k )
    assert isinstance( result, int )
    assert result >= 1

  # --- Symmetry C(N, k) == C(N, N-k) ---
  def test_symmetry( self ) -> None:
    '''Verify symmetry C(N, k) == C(N, N-k).'''
    for n in [ 0, 1, 10, 30 ]:
      for k in range( n + 1 ): assert Combinations( n, k ) == Combinations( n, n - k )
