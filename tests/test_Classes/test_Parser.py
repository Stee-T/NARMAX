import pytest
from typing import Optional
from NARMAX.Classes.Parser_0_3 import ExpressionParser, ExprNode

# Helper to quickly build expected ASTs
def L( Coeff: Optional[ float ] = None, Var: Optional[ str ] = None, Lag: Optional[ int ] = None, Exp: Optional[ float ] = None ) -> ExprNode:
  return ExprNode( Coeff = Coeff, VarName = Var, Lag = Lag, Exponent = Exp )


def B( Func: Optional[ str ] = None, Subs: Optional[ list[ ExprNode ] ] = None, Ops: Optional[ list[ str ] ] = None, Coeff: Optional[ float ] = None ) -> ExprNode:
  return ExprNode( FuncName = Func, SubExpressions = Subs or [], Operators = Ops or [], Coeff = Coeff )

# ---------------------------------------------------------------------------
# Helper functions for structural invariant checks (used to harden all tests)
# ---------------------------------------------------------------------------
def _check_node_fields( node: ExprNode ) -> None:
  '''Recursively verify that every ExprNode in the tree has valid field types.'''
  assert isinstance( node, ExprNode )
  assert ( node.Coeff is None ) or ( isinstance( node.Coeff, ( int, float ) ) )
  assert ( node.VarName is None ) or ( isinstance( node.VarName, str ) )
  assert ( node.Lag is None ) or ( isinstance( node.Lag, int ) )
  assert ( node.Exponent is None ) or ( isinstance( node.Exponent, ( int, float ) ) )
  assert ( node.FuncName is None ) or ( isinstance( node.FuncName, str ) )
  assert isinstance( node.SubExpressions, list )
  assert isinstance( node.Operators, list )
  for sub in node.SubExpressions: _check_node_fields( sub )


def _check_ast( result: ExprNode ) -> None:
  '''Type + structural invariant check for a parser result.'''
  assert isinstance( result, ExprNode )
  _check_node_fields( result )


def _check_equal_asts( expr: str, result: ExprNode ) -> None:
  '''Two parses of the same expression produce equal ASTs.'''
  assert ExpressionParser( expr ) == result

# Expected error message fragments for invalid expression cases.
_INVALID_MSGS = {
  "fractional lag": "contains fractional lags",
  "trailing operator": "Unexpected end of expression",
  "malformed number": "Invalid numeric coefficient",
  "space in function name": "contains a function name with spaces",
  "empty string": "Empty expression",
  "whitespace only": "Empty expression",
  "empty function arguments": "Invalid expression structure",
  "invalid trailing character": "Unexpected character: #",
  "fractional lag leading dot": "contains fractional lags",
}

TEST_CASES = [
  ( "sin(0.957*x[k-1]^3)", B( None, [ B( 'sin', [ L( 0.957 ), L( Var = 'x', Lag = -1, Exp = 3.0 ) ], [ '*' ] ) ], [] ), "function with coefficient * argument ^Exponent" ),
  ( "0.6x1[k+2]^11", B( None, [ L( 0.6 ), L( Var = 'x1', Lag = 2, Exp = 11.0 ) ], [ '*' ] ), "no function, implicit mul, positive lag, double digit exponent" ),
  ( "-0.6x9^2", B( None, [ L( -0.6 ), L( Var = 'x9', Exp = 2.0 ) ], [ '*' ] ), "no function, no lag" ),
  ( "-0.5x[k-11] ^ 3*y[k-1]", B( None, [ L( -0.5 ), L( Var = 'x', Lag = -11, Exp = 3.0 ), L( Var = 'y', Lag = -1 ) ], [ '*', '*' ] ), "exponent and multiplication" ),
  ( "exp(0.8*x_2[k-4] - 0.6*x_1[k]^2 * y[k-2])", B( None, [ B( 'exp', [ L( 0.8 ), L( Var = 'x_2', Lag = -4 ), L( -0.6 ), L( Var = 'x_1', Exp = 2.0 ), L( Var = 'y', Lag = -2 ) ], [ '*', '+', '*', '*' ] ) ], [] ), "complex nested" ),
  ( "3*sin( cos(2*x[k] + 5) - 4*y[k-1]^3 )", B( None, [ L( 3.0 ), B( 'sin', [ B( 'cos', [ L( 2.0 ), L( Var = 'x' ), L( 5.0 ) ], [ '*', '+' ] ), L( -4.0 ), L( Var = 'y', Lag = -1, Exp = 3.0 ) ], [ '+', '*' ] ) ], [ '*' ] ), "deeply nested user example" ),
  ( "x * (y + z)", B( None, [ L( Var = 'x' ), B( None, [ L( Var = 'y' ), L( Var = 'z' ) ], [ '+' ] ) ], [ '*' ] ), "parentheses grouping precedence" ),
  ( "a - (b - c)", B( None, [ L( Var = 'a' ), B( None, [ L( Var = 'b' ), L( -1.0, Var = 'c' ) ], [ '+' ], Coeff = -1.0 ) ], [ '+' ] ), "nested subtraction" ),
  ( "~/abs(x1[k-1] + 0.8*x2[k-4])", B( None, [ B( '~/abs', [ L( Var = 'x1', Lag = -1 ), L( 0.8 ), L( Var = 'x2', Lag = -4 ) ], [ '+', '*' ] ) ], [] ), "rational non-lin denominator" ),
  ( "~/(x1[k-1] + 0.8*x2[k-4])", B( None, [ B( '~/', [ L( Var = 'x1', Lag = -1 ), L( 0.8 ), L( Var = 'x2', Lag = -4 ) ], [ '+', '*' ] ) ], [] ), "bare ~/ prefix without function name" ),
  ( "1/(x[k-1])", B( None, [ B( '1/', [ L( Var = 'x', Lag = -1 ) ], [] ) ], [] ), "bare 1/ prefix without function name" ),
  ( "1/abs(0.2*x[k] + 0.8*y2[j-4])", B( None, [ B( '1/abs', [ L( 0.2 ), L( Var = 'x' ), L( 0.8 ), L( Var = 'y2', Lag = -4 ) ], [ '*', '+', '*' ] ) ], [] ), "1/ prefix with function name" ),
  ( "sin(x) + y", B( None, [ B( 'sin', [ L( Var = 'x' ) ] ), L( Var = 'y' ) ], [ '+' ] ), "valid function call followed by expression" ),
  ( "-y[k-2] - 0.6x1[k-2]^3", B( None, [ L( -1.0, Var = 'y', Lag = -2 ), L( -0.6 ), L( Var = 'x1', Lag = -2, Exp = 3.0 ) ], [ '+', '*' ] ), "subtraction merging, implied -1 coeff, implicit mul" ),
  ( "sin( 0.8 * x1[k-1]^3 - 0.2 )", B( None, [ B( 'sin', [ L( 0.8 ), L( Var = 'x1', Lag = -1, Exp = 3.0 ), L( -0.2 ) ], [ '*', '+' ] ) ], [] ), "scalar argument inside function" ),
  ( "0.9*x1[k-1]^3 / 0.6*x1[k-2]^3", B( None, [ L( 0.9 ), L( Var = 'x1', Lag = -1, Exp = 3.0 ), L( 0.6 ), L( Var = 'x1', Lag = -2, Exp = 3.0 ) ], [ '*', '/', '*' ] ), "fraction operator between terms" ),
  ( "0.9*x1^3 / 0.679 * x1[k-2]^3.5", B( None, [ L( 0.9 ), L( Var = 'x1', Exp = 3.0 ), L( 0.679 ), L( Var = 'x1', Lag = -2, Exp = 3.5 ) ], [ '*', '/', '*' ] ), "fractions no lag, continued multiplication" ),
  ( "32*sin(0.9*x1[k-1]^3 + 0.6x1[k-2]^3)", B( None, [ L( 32.0 ), B( 'sin', [ L( 0.9 ), L( Var = 'x1', Lag = -1, Exp = 3.0 ), L( 0.6 ), L( Var = 'x1', Lag = -2, Exp = 3.0 ) ], [ '*', '+', '*' ] ) ], [ '*' ] ), "coefficient times function call" ),
]

@pytest.mark.parametrize( "expr, expected, comment", TEST_CASES, ids = [ c[ 2 ] for c in TEST_CASES ] )
def test_parser_valid_expressions( expr: str, expected: ExprNode, comment: str ) -> None:
  result: ExprNode = ExpressionParser( expr )
  assert result == expected
  _check_ast( result )
  _check_equal_asts( expr, result )

INVALID_CASES: list[ tuple[ str, str ] ] = [
  ( "x1[k-1.5]^3", "fractional lag" ),
  ( "x +", "trailing operator" ),
  ( "1.2.3*x", "malformed number" ),
  ( "my func(x)", "space in function name" ),
  ( "", "empty string" ),
  ( "   ", "whitespace only" ),
  ( "func()", "empty function arguments" ),
  ( "x[k-1]#", "invalid trailing character" ),
  ( "x[k-.5]", "fractional lag leading dot" ),
]

@pytest.mark.parametrize( "expr, comment", INVALID_CASES, ids = [ c[ 1 ] for c in INVALID_CASES ] )
def test_parser_invalid_expressions( expr: str, comment: str ) -> None:
  with pytest.raises( ValueError, match = _INVALID_MSGS[ comment ] ):
    ExpressionParser( expr )


def test_implicit_multiplication() -> None:
  res: ExprNode = ExpressionParser( "2x(3y)" )
  assert res.Operators == [ '*' ]
  assert len( res.SubExpressions ) == 2
  assert res.SubExpressions[ 1 ].FuncName == 'x'
  _check_ast( res )
  _check_equal_asts( "2x(3y)", res )


def test_unary_minus_handling() -> None:
  res: ExprNode = ExpressionParser( "-x - -y" )
  assert res.SubExpressions[ 0 ].Coeff == -1.0
  assert res.SubExpressions[ 1 ].Coeff is None
  assert res.Operators == [ '+' ]
  _check_ast( res )
  _check_equal_asts( "-x - -y", res )


def test_reserved_sequence_lag() -> None:
  with pytest.raises( ValueError, match = "__LAG" ):
    ExpressionParser( "x__LAG0__[k-1]" )


def test_non_string_input() -> None:
  with pytest.raises( ValueError, match = "not a string" ):
    ExpressionParser( 123 )


def test_simple_multiplication() -> None:
  res: ExprNode = ExpressionParser( "x * y" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 1 ].VarName == 'y'
  assert res.Operators == [ '*' ]
  _check_ast( res )
  _check_equal_asts( "x * y", res )


def test_implicit_lag_zero() -> None:
  res: ExprNode = ExpressionParser( "x[k]" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 0 ].Lag is None
  _check_ast( res )
  _check_equal_asts( "x[k]", res )


def test_trailing_space_in_function_name() -> None:
  res: ExprNode = ExpressionParser( "sin(x[k])" )
  assert res.SubExpressions[ 0 ].FuncName == "sin"
  _check_ast( res )
  _check_equal_asts( "sin(x[k])", res )


def test_subtraction_merging_creates_spurious_operator() -> None:
  res: ExprNode = ExpressionParser( "x*-5" )
  assert res.Operators == [ '*' ]
  assert len( res.SubExpressions ) == 2
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 1 ].Coeff == -5.0
  _check_ast( res )
  _check_equal_asts( "x*-5", res )


def test_unspaced_minus_swallows_second_term() -> None:
  res: ExprNode = ExpressionParser( "x-2" )
  assert len( res.SubExpressions ) == 2
  assert res.Operators == [ '+' ]
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 1 ].Coeff == -2.0
  _check_ast( res )
  _check_equal_asts( "x-2", res )


def test_missing_operator_between_variables() -> None:
  with pytest.raises( ValueError, match = "Unexpected token at end" ):
    ExpressionParser( "x y" )


def test_unspaced_plus() -> None:
  res: ExprNode = ExpressionParser( "x+2" )
  assert len( res.SubExpressions ) == 2
  assert res.Operators == [ '+' ]
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 1 ].Coeff == 2.0
  assert res.SubExpressions[ 1 ].VarName is None
  _check_ast( res )
  _check_equal_asts( "x+2", res )


def test_division_by_negative_constant() -> None:
  res: ExprNode = ExpressionParser( "x / -5" )
  assert res.Operators == [ '/' ]
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 0 ].Coeff is None
  assert res.SubExpressions[ 1 ].Coeff == -5.0
  assert res.SubExpressions[ 1 ].VarName is None
  _check_ast( res )
  _check_equal_asts( "x / -5", res )


def test_spaces_inside_lag_brackets() -> None:
  res: ExprNode = ExpressionParser( "x[ k - 1 ]" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 0 ].Lag == -1
  _check_ast( res )
  _check_equal_asts( "x[ k - 1 ]", res )


class TestParser:
  def test_simple_constant( self ) -> None:
    res: ExprNode = ExpressionParser( "3.5" )
    assert len( res.SubExpressions ) == 1
    assert res.SubExpressions[ 0 ].Coeff == 3.5
    assert res.SubExpressions[ 0 ].VarName is None
    assert res.SubExpressions[ 0 ].Lag is None
    assert res.SubExpressions[ 0 ].Exponent is None
    _check_ast( res )
    _check_equal_asts( "3.5", res )

  def test_variable_no_lag( self ) -> None:
    res: ExprNode = ExpressionParser( "x1" )
    assert len( res.SubExpressions ) == 1
    assert res.SubExpressions[ 0 ].VarName == "x1"
    assert res.SubExpressions[ 0 ].Lag is None
    assert res.SubExpressions[ 0 ].Coeff is None
    _check_ast( res )
    _check_equal_asts( "x1", res )

  def test_coefficient_and_lag( self ) -> None:
    res: ExprNode = ExpressionParser( "2.0*x1[k-1]" )
    assert res.Operators == [ '*' ]
    se: ExprNode = res.SubExpressions[ 0 ]
    assert se.Coeff == 2.0
    se = res.SubExpressions[ 1 ]
    assert se.VarName == "x1"
    assert se.Lag == -1
    _check_ast( res )
    _check_equal_asts( "2.0*x1[k-1]", res )

  def test_negative_coefficient( self ) -> None:
    res: ExprNode = ExpressionParser( "-x1" )
    assert len( res.SubExpressions ) == 1
    assert res.SubExpressions[ 0 ].Coeff == -1.0
    assert res.SubExpressions[ 0 ].VarName == "x1"
    _check_ast( res )
    _check_equal_asts( "-x1", res )

  def test_exponent( self ) -> None:
    res: ExprNode = ExpressionParser( "x1^2" )
    assert len( res.SubExpressions ) == 1
    assert res.SubExpressions[ 0 ].Exponent == 2.0
    assert res.SubExpressions[ 0 ].VarName == "x1"
    assert res.SubExpressions[ 0 ].Coeff is None
    _check_ast( res )
    _check_equal_asts( "x1^2", res )

  def test_function_name( self ) -> None:
    res: ExprNode = ExpressionParser( "myFunc(x1[k])" )
    assert res.SubExpressions[ 0 ].FuncName == "myFunc"
    assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x1"
    _check_ast( res )
    _check_equal_asts( "myFunc(x1[k])", res )

  def test_rational_prefix( self ) -> None:
    res: ExprNode = ExpressionParser( "~/myFunc(x1)" )
    assert res.SubExpressions[ 0 ].FuncName == "~/myFunc"
    _check_ast( res )
    _check_equal_asts( "~/myFunc(x1)", res )

  def test_multiple_terms( self ) -> None:
    res: ExprNode = ExpressionParser( "x1 + 2.0*x2[k-1] - 3.5*x3^2" )
    assert len( res.SubExpressions ) == 5
    assert res.SubExpressions[ 0 ].VarName == 'x1'
    assert res.SubExpressions[ 1 ].Coeff == 2.0
    assert ( res.SubExpressions[ 2 ].VarName == 'x2' ) and ( res.SubExpressions[ 2 ].Lag == -1 )
    assert res.SubExpressions[ 3 ].Coeff == -3.5
    assert ( res.SubExpressions[ 4 ].VarName == 'x3' ) and ( res.SubExpressions[ 4 ].Exponent == 2.0 )
    assert res.Operators == [ '+', '*', '+', '*' ]
    _check_ast( res )
    _check_equal_asts( "x1 + 2.0*x2[k-1] - 3.5*x3^2", res )

  def test_1_over_function( self ) -> None:
    res: ExprNode = ExpressionParser( "1/myFunc(x1)" )
    assert res.SubExpressions[ 0 ].FuncName == "1/myFunc"
    _check_ast( res )
    _check_equal_asts( "1/myFunc(x1)", res )

def test_tilde_slash_with_extra_spaces() -> None:
  res: ExprNode = ExpressionParser( "~  / (x[k])" )
  assert res.SubExpressions[ 0 ].FuncName == "~/"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].Lag is None
  _check_ast( res )
  _check_equal_asts( "~  / (x[k])", res )


def test_one_over_with_extra_spaces() -> None:
  res: ExprNode = ExpressionParser( "1 / (x[k])" )
  assert res.SubExpressions[ 0 ].FuncName == "1/"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x"
  _check_ast( res )
  _check_equal_asts( "1 / (x[k])", res )


def test_one_over_with_multiple_spaces() -> None:
  res: ExprNode = ExpressionParser( "1  / (x[k])" )
  assert res.SubExpressions[ 0 ].FuncName == "1/"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x"
  _check_ast( res )
  _check_equal_asts( "1  / (x[k])", res )


def test_variable_starting_with_digit() -> None:
  res: ExprNode = ExpressionParser( "1x" )
  assert res.SubExpressions[ 0 ].Coeff == 1.0
  assert res.SubExpressions[ 1 ].VarName == "x"
  assert res.Operators == [ '*' ]
  _check_ast( res )
  _check_equal_asts( "1x", res )


def test_variable_as_exponent() -> None:
  res: ExprNode = ExpressionParser( "x ^ y" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 1 ].VarName == 'y'
  assert res.Operators == [ '^' ]
  _check_ast( res )
  _check_equal_asts( "x ^ y", res )


def test_constant_with_exponent() -> None:
  res: ExprNode = ExpressionParser( "3.5^2" )
  assert res.SubExpressions[ 0 ].Coeff == 3.5
  assert res.SubExpressions[ 1 ].Coeff == 2.0
  assert res.Operators == [ '^' ]
  _check_ast( res )
  _check_equal_asts( "3.5^2", res )


def test_negative_constant_with_exponent() -> None:
  res: ExprNode = ExpressionParser( "-2^3" )
  assert res.SubExpressions[ 0 ].Coeff == -2.0
  assert res.SubExpressions[ 1 ].Coeff == 3.0
  assert res.Operators == [ '^' ]
  _check_ast( res )
  _check_equal_asts( "-2^3", res )


def test_leading_plus() -> None:
  res: ExprNode = ExpressionParser( "+x" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 0 ].Coeff is None
  _check_ast( res )
  _check_equal_asts( "+x", res )


def test_chained_exponents() -> None:
  # First ^2 consumed as variable exponent, second ^3 is binary operator
  res: ExprNode = ExpressionParser( "x ^ 2 ^ 3" )
  assert res.SubExpressions[ 0 ].VarName == 'x'
  assert res.SubExpressions[ 0 ].Exponent == 2.0
  assert res.SubExpressions[ 1 ].Coeff == 3.0
  assert res.Operators == [ '^' ]
  _check_ast( res )
  _check_equal_asts( "x ^ 2 ^ 3", res )


def test_nested_function_call() -> None:
  res: ExprNode = ExpressionParser( "f(g(x))" )
  assert res.SubExpressions[ 0 ].FuncName == "f"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].FuncName == "g"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x"
  _check_ast( res )
  _check_equal_asts( "f(g(x))", res )


def test_unary_minus_on_function() -> None:
  res: ExprNode = ExpressionParser( "-sin(x)" )
  assert res.SubExpressions[ 0 ].Coeff == -1.0
  assert res.SubExpressions[ 0 ].FuncName == "sin"
  _check_ast( res )
  _check_equal_asts( "-sin(x)", res )


def test_division_of_functions() -> None:
  res: ExprNode = ExpressionParser( "sin(x) / cos(y)" )
  assert len( res.SubExpressions ) == 2
  assert res.SubExpressions[ 0 ].FuncName == "sin"
  assert res.SubExpressions[ 1 ].FuncName == "cos"
  assert res.Operators == [ '/' ]
  _check_ast( res )
  _check_equal_asts( "sin(x) / cos(y)", res )


def test_deep_three_level_nesting() -> None:
  res: ExprNode = ExpressionParser( "f(g(h(x)))" )
  assert res.SubExpressions[ 0 ].FuncName == "f"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].FuncName == "g"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].SubExpressions[ 0 ].FuncName == "h"
  assert res.SubExpressions[ 0 ].SubExpressions[ 0 ].SubExpressions[ 0 ].SubExpressions[ 0 ].VarName == "x"
  _check_ast( res )
  _check_equal_asts( "f(g(h(x)))", res )


def test_leading_minus_function_term() -> None:
  res: ExprNode = ExpressionParser( "-sin(x) + cos(y)" )
  assert res.SubExpressions[ 0 ].Coeff == -1.0
  assert res.SubExpressions[ 0 ].FuncName == "sin"
  assert res.SubExpressions[ 1 ].FuncName == "cos"
  assert res.Operators == [ '+' ]
  _check_ast( res )
  _check_equal_asts( "-sin(x) + cos(y)", res )
