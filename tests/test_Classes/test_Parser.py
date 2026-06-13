# tests/test_Classes/test_Parser.py
import pytest
from NARMAX.Classes.Parser_0_2 import ExpressionParser, ParsedReg, SubExpression

# Each tuple: ( expression_string, expected_ParsedReg, comment )
TEST_CASES = [
  (
    "sin(0.957*x[k-1]^3)",
    ParsedReg( 'sin', [ SubExpression( 0.957, 'x', -1, 3.0 ) ], [] ),
    "function with coefficient * argument ^Exponent"
   ),
  (
    "0.6x1[k+2]^11",
    ParsedReg( None, [ SubExpression( 0.6, 'x1', 2, 11.0 ) ], [] ),
    "no function, no multiplication sign, number in varname, positive lag, double digit exponent"
   ),
  (
    "-0.6x9^2",
    ParsedReg( None, [ SubExpression( -0.6, 'x9', None, 2.0 ) ], [] ),
    "no function, no multiplication sign, no time lag"
   ),
  (
    "-0.5x[k-11] ^ 3*y[k-1]",
    ParsedReg( None, [
      SubExpression( -0.5, 'x', -11, None ),
      SubExpression( 3.0, 'y', -1, None )
    ], [ '^' ] ),
    "double digit lag, exponent with another expression"
   ),
  (
    "exp(0.8*x_2[k-4] - 0.6*x_1[k]^2 * y[k-2])",
    ParsedReg( 'exp', [
      SubExpression( 0.8, 'x_2', -4, None ),
      SubExpression( -0.6, 'x_1', None, 2.0 ),
      SubExpression( None, 'y', -2, None )
    ], [ '+', '*' ] ),
    "function, multiple arguments and coefficients, underscores in variable names"
   ),
  (
    "-y[k-2] - 0.6x1[k-2]^3",
    ParsedReg( None, [
      SubExpression( -1.0, 'y', -2, None ),
      SubExpression( -0.6, 'x1', -2, 3.0 )
    ], [ '+' ] ),
    "Minus without coefficient, minus not part of coefficient"
   ),
  (
    "sin( 0.8 * x1[k-1]^3 - 0.2 )",
    ParsedReg( 'sin', [
      SubExpression( 0.8, 'x1', -1, 3.0 ),
      SubExpression( -0.2, None, None, None )
    ], [ '+' ] ),
    "Scalar argument, random spaces ( one of each making 0.8 a standalone Subexpression)"
   ),
  (
    "~/(x1[k-1] + 0.8* x2[k-4])",
    ParsedReg( '~/', [
      SubExpression( None, 'x1', -1, None ),
      SubExpression( 0.8, 'x2', -4, None )
    ], [ '+' ] ),
    "Denominator terms: ~/ Expr, unnecessary space between coeff and expression"
   ),
  (
    "0.9*x1[k-1]^3 / 0.6*x1[k-2]^3",
    ParsedReg( None, [
      SubExpression( 0.9, 'x1', -1, 3.0 ),
      SubExpression( 0.6, 'x1', -2, 3.0 )
    ], [ '/' ] ),
    "Fractions"
   ),
  (
    "0.9*x1^3 / 0.679 * x1[k-2]^3.5",
    ParsedReg( None, [
      SubExpression( 0.9, 'x1', None, 3.0 ),
      SubExpression( 0.679, 'x1', -2, 3.5 )
    ], [ '/' ] ),
    "Fractions, no lag, unnecessary spaces ( see comment below on how spaces imply parenthesis)"
   ),
  (
    "~/abs(x1[k-1] + 0.8*x2[k-4])",
    ParsedReg( '~/abs', [
      SubExpression( None, 'x1', -1, None ),
      SubExpression( 0.8, 'x2', -4, None )
    ], [ '+' ] ),
    "Parenthesis + non-lin denominator term"
   ),
  (
    "1/abs(0.2*x[k] + 0.8*y2[j-4])",
    ParsedReg( '1/abs', [
      SubExpression( 0.2, 'x', None, None ),
      SubExpression( 0.8, 'y2', -4, None )
    ], [ '+' ] ),
    "Parenthesis + actual fraction with non-lin, other lag variable"
   ),
  (
    "32*sin(0.9*x1[k-1]^3 + 0.6x1[k-2]^3)",
    ParsedReg( '32*sin', [
      SubExpression( 0.9, 'x1', -1, 3.0 ),
      SubExpression( 0.6, 'x1', -2, 3.0 )
    ], [ '+' ] ),
    "Weird but user's problem: 32* is part of the func name, lol"
   ),
]

# Generate explicit test IDs from the comments
TEST_IDS = [ f"{ expr }  # { comment }" for expr, _, comment in TEST_CASES ]

# Invalid expressions that should raise an exception
INVALID_CASES = [
  ( "x1[k-1.5]^3", "decimal lag correctly raises an error" ),
  ( "x[k-.5]", "decimal lag without leading digit" ),
  ( "func()", "empty arguments" ),
  ( "x +", "trailing operator" ),
  ( "x[k-1]#", "invalid trailing characters (fullmatch enforcement)" ),
  ( "sin(x) + y", "multiple parentheses / expression after closing paren" ),
]
INVALID_IDS = [ f"{ expr }  # { comment }" for expr, comment in INVALID_CASES ]

FRACTIONAL_LAG_BYPASS_CASES = [
  ( "x[k-.5]", "fractional lag, no leading digit (minus)" ),
  ( "y[k+.25]", "fractional lag, no leading digit (plus)" ),
  ( "0.8*z[j-.1]^2", "fractional lag inside coefficient/expression" ),
  ( "sin(a[k-.9] + b[k])", "fractional lag inside function call" ),
]
FRACTIONAL_LAG_BYPASS_IDS = [ f"{ expr }  # { comment }" for expr, comment in FRACTIONAL_LAG_BYPASS_CASES ]

@pytest.mark.parametrize(
  "expr, expected, comment",
  TEST_CASES,
  ids = TEST_IDS
 )
def test_parser_valid_expressions( expr, expected, comment ) -> None:
  '''Test that ExpressionParser correctly parses valid expressions.'''
  result = ExpressionParser( expr )
  assert result == expected, f"Parsing failed for: { expr } ( { comment })"


@pytest.mark.parametrize(
  "expr, comment",
  INVALID_CASES,
  ids = INVALID_IDS
 )
def test_parser_invalid_expressions( expr, comment ) -> None:
  '''Test that ExpressionParser raises ValueError for invalid expressions.'''
  with pytest.raises( ValueError ): ExpressionParser( expr )


@pytest.mark.parametrize(
  "expr, comment",
  FRACTIONAL_LAG_BYPASS_CASES,
  ids = FRACTIONAL_LAG_BYPASS_IDS
 )
def test_parser_fractional_lag_bypass( expr, comment ) -> None:
  '''Verify fractional lags that bypass the original regex are caught early with the correct error.'''
  with pytest.raises( ValueError, match = r"contains fractional lags" ):
    ExpressionParser( expr )

    # Additional tests that expose parser bugs and missing edge cases


def test_trailing_space_in_function_name() -> None:
  '''Bug 1: FuncName retains a trailing space, e.g. 'sin ' instead of 'sin'.'''
  result = ExpressionParser( "sin(x[k])" )
  assert result.FuncName == "sin", f"Expected 'sin', got '{ result.FuncName }'"
  assert result.SubExpressions == [ SubExpression( None, 'x', None, None ) ]
  assert result.Operators == []


def test_subtraction_merging_creates_spurious_operator() -> None:
  ''''x*-5' introduces a false '+' operator after cleaning.'''
  result = ExpressionParser( "x*-5" )
  # Correct behaviour: two subexpressions with a single '*' operator.
  # The bug makes operators = ['*', '+'].
  assert result.FuncName is None
  assert result.Operators == [ '*' ], f"Expected ['*'], got { result.Operators }"
  assert len( result.SubExpressions ) == 2
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Coeff is None
  assert result.SubExpressions[ 1 ].Coeff == -5.0 # the constant -5
  assert result.SubExpressions[ 1 ].VarName is None


def test_unspaced_minus_swallows_second_term() -> None:
  ''''x-2' without spaces now correctly splits and merges subtraction.'''
  result = ExpressionParser( "x-2" )
  assert result.FuncName is None
  assert len( result.SubExpressions ) == 2
  assert result.Operators == [ '+' ]
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Coeff is None
  assert result.SubExpressions[ 1 ].Coeff == -2.0
  assert result.SubExpressions[ 1 ].VarName is None


def test_malformed_number_with_multiple_dots() -> None:
  ''''1.2.3*x' causes an uncaught float conversion error.'''
  with pytest.raises( ValueError, match = "Invalid numeric coefficient" ):
    ExpressionParser( "1.2.3*x" )


def test_tilde_slash_with_extra_spaces() -> None:
  ''''~  / (args)' fails to restore the '~/' operator name.'''
  result = ExpressionParser( "~  / (x[k])" )
  # Without the fix, FuncName becomes '~' instead of '~/'.
  assert result.FuncName == "~/", f"Expected '~/', got '{ result.FuncName }'"
  assert result.SubExpressions == [ SubExpression( None, 'x', None, None ) ]
  assert result.Operators == []


def test_empty_input() -> None:
  '''Edge case: empty string should be rejected gracefully.'''
  with pytest.raises( Exception ):
    ExpressionParser( "" )


def test_whitespace_only_input() -> None:
  '''Edge case: whitespace-only input should be rejected.'''
  with pytest.raises( Exception ):
    ExpressionParser( "   " )


def test_missing_operator_between_variables() -> None:
  '''Edge case: 'x y' with no operator should fail (mismatch error).'''
  with pytest.raises( ValueError, match = "Mismatch between the number of expressions" ):
    ExpressionParser( "x y" )


def test_variable_starting_with_digit() -> None:
  '''
  Edge case: '1x' is currently parsed as '1*x'.
  Include this test to document the behaviour – change if it becomes illegal.
  '''
  result = ExpressionParser( "1x" )
  assert result.FuncName is None
  assert result.Operators == []
  assert result.SubExpressions == [ SubExpression( 1.0, 'x', None, None ) ]


def test_negative_constant_with_exponent() -> None:
  '''
  Edge case: '-2^3' is parsed as a constant with an exponent,
  which the parser deliberately rejects. Verify the error.
  '''
  with pytest.raises( ValueError, match = "Exponent without variable" ):
    ExpressionParser( "-2^3" )


# Additional edge‑case tests (complete the missing cases)

def test_one_over_with_extra_spaces() -> None:
  ''''1 / (x)' with extra spaces should restore function name '1/'.'''
  result = ExpressionParser( "1 / (x[k])" )
  assert result.FuncName == "1/", f"Expected '1/', got '{ result.FuncName }'"
  assert result.SubExpressions == [ SubExpression( None, 'x', None, None ) ]
  assert result.Operators == []


def test_one_over_with_multiple_spaces() -> None:
  ''''1  / (x)' (two spaces) should still restore '1/'.'''
  result = ExpressionParser( "1  / (x[k])" )
  assert result.FuncName == "1/", f"Expected '1/', got '{ result.FuncName }'"
  assert result.SubExpressions == [ SubExpression( None, 'x', None, None ) ]
  assert result.Operators == []


def test_unspaced_plus() -> None:
  ''''x+2' without spaces must split into two subexpressions with a '+' operator.'''
  result = ExpressionParser( "x+2" )
  assert result.FuncName is None
  assert len( result.SubExpressions ) == 2, (
        f"Expected 2 subexpressions for 'x+2', got { len( result.SubExpressions ) }"
    )
  assert result.Operators == [ '+' ]
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Coeff is None
  assert result.SubExpressions[ 1 ].Coeff == 2.0
  assert result.SubExpressions[ 1 ].VarName is None


def test_division_by_negative_constant() -> None:
  ''''x / -5' should have one '/' operator and constant -5, no spurious '+'.'''
  result = ExpressionParser( "x / -5" )
  assert result.FuncName is None
  assert result.Operators == [ '/' ]
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Coeff is None
  assert result.SubExpressions[ 1 ].Coeff == -5.0
  assert result.SubExpressions[ 1 ].VarName is None


def test_chained_exponents() -> None:
  ''''x ^ 2 ^ 3' is parsed as three subexpressions with two '^' operators.'''
  result = ExpressionParser( "x ^ 2 ^ 3" )
  assert result.SubExpressions == [
        SubExpression( None, 'x', None, None ),
        SubExpression( 2.0, None, None, None ),
        SubExpression( 3.0, None, None, None )
    ]
  assert result.Operators == [ '^', '^' ]


def test_function_name_with_space() -> None:
  ''''my func(x)' contains a space in the function name and should be rejected.'''
  with pytest.raises( ValueError, match = "contains a function name with spaces" ):
    ExpressionParser( "my func(x)" )


class TestParser:
  def test_simple_constant( self ) -> None:
    '''Parsing a simple numeric constant.'''
    expr = "3.5"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    assert res.SubExpressions[ 0 ].Coeff == 3.5
    assert res.SubExpressions[ 0 ].VarName is None
    assert res.SubExpressions[ 0 ].Lag is None
    assert res.SubExpressions[ 0 ].Exponent is None

  def test_variable_no_lag( self ) -> None:
    '''Parsing a variable name with no lag.'''
    expr = "x1"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.VarName == "x1"
    assert se.Lag is None # interpreted as [k]
    assert se.Coeff is None
    assert se.Exponent is None

  def test_coefficient_and_lag( self ) -> None:
    '''Parsing a coefficient, variable, and lag together.'''
    expr = "2.0*x1[k-1]"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert res.Operators == []
    se = res.SubExpressions[ 0 ]
    assert se.Coeff == 2.0
    assert se.VarName == "x1"
    assert se.Lag == -1
    assert se.Exponent is None

  def test_negative_coefficient( self ) -> None:
    '''Parsing a negative coefficient (implied -1).'''
    expr = "-x1"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.Coeff == -1.0
    assert se.VarName == "x1"
    assert se.Lag is None
    assert se.Exponent is None

  def test_exponent( self ) -> None:
    '''Parsing an exponent on a variable.'''
    expr = "x1^2"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.Exponent == 2.0
    assert se.VarName == "x1"
    assert se.Coeff is None
    assert se.Lag is None

  def test_function_name( self ) -> None:
    '''Parsing a function name with parentheses.'''
    expr = "myFunc(x1[k])"
    res = ExpressionParser( expr )
    assert res.FuncName == "myFunc"
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.VarName == "x1"
    assert se.Lag is None
    assert se.Coeff is None

  def test_rational_prefix( self ) -> None:
    '''Parsing the rational prefix ~/ on a function.'''
    expr = "~/myFunc(x1)"
    res = ExpressionParser( expr )
    assert res.FuncName == "~/myFunc"
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.VarName == "x1"
    assert se.Coeff is None

  def test_1_over_function( self ) -> None:
    '''Parsing the 1/ prefix on a function.'''
    expr = "1/myFunc(x1)"
    res = ExpressionParser( expr )
    assert res.FuncName == "1/myFunc"
    assert res.Operators == []
    assert len( res.SubExpressions ) == 1
    se = res.SubExpressions[ 0 ]
    assert se.VarName == "x1"
    assert se.Coeff is None

  def test_multiple_terms( self ) -> None:
    '''Parsing an expression with multiple terms and operators.'''
    expr = "x1 + 2.0*x2[k-1] - 3.5*x3^2"
    res = ExpressionParser( expr )
    assert res.FuncName is None
    assert len( res.SubExpressions ) == 3
    # subtraction is merged into the coefficient, so operator is always '+'
    assert res.Operators == [ '+', '+' ]

  def test_malformed_number_multiple_dots( self ) -> None:
    '''A number with multiple decimal dots raises ValueError.'''
    with pytest.raises( ValueError, match = "Invalid numeric coefficient" ):
      ExpressionParser( "1.2.3*x1" )

  def test_spaces_inside_lag_brackets( self ) -> None:
    '''Spaces inside lag brackets should be rejected.'''
    with pytest.raises( ValueError, match = "not recognized as a valid expression" ):
      ExpressionParser( "x1[ k - 1 ]" )

  def test_fractional_lag_raises( self ) -> None:
    '''A fractional lag raises the correct error.'''
    with pytest.raises( ValueError, match = "fractional lags" ):
      ExpressionParser( "x1[k+0.5]" )

  def test_multiple_parentheses_raises( self ) -> None:
    '''Nested/multiple parentheses raise an error.'''
    with pytest.raises( ValueError, match = "multiple parentheses" ):
      ExpressionParser( "f(g(x))" )


# Additional edge‑case tests for uncovered scenarios

def test_reserved_sequence_lag() -> None:
  '''Input containing the reserved sequence __LAG should be rejected.'''
  with pytest.raises( ValueError, match = "__LAG" ):
    ExpressionParser( "x__LAG0__[k-1]" )


def test_non_string_input() -> None:
  '''Passing a non-string to ExpressionParser should raise TypeError or ValueError.'''
  with pytest.raises( ValueError, match = "not a string" ):
    ExpressionParser( 123 )


def test_simple_multiplication() -> None:
  ''''x * y' parses as two subexpressions with a '*' operator.'''
  result = ExpressionParser( "x * y" )
  assert result.FuncName is None
  assert result.Operators == [ '*' ]
  assert len( result.SubExpressions ) == 2
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Coeff is None
  assert result.SubExpressions[ 1 ].VarName == 'y'
  assert result.SubExpressions[ 1 ].Coeff is None


def test_variable_as_exponent() -> None:
  ''''x ^ y' uses variable y as the exponent value.'''
  result = ExpressionParser( "x ^ y" )
  assert result.FuncName is None
  assert result.Operators == [ '^' ]
  assert len( result.SubExpressions ) == 2
  assert result.SubExpressions[ 0 ].VarName == 'x'
  assert result.SubExpressions[ 0 ].Exponent is None
  assert result.SubExpressions[ 1 ].VarName == 'y'


def test_constant_with_exponent_raises() -> None:
  '''A positive constant raised to an exponent raises 'Exponent without variable'.'''
  with pytest.raises( ValueError, match = "Exponent without variable" ):
    ExpressionParser( "3.5^2" )


def test_implicit_lag_zero() -> None:
  ''''x[k]' with no +/- lag value is parsed as lag=None (zero shift).'''
  result = ExpressionParser( "x[k]" )
  assert result.SubExpressions == [ SubExpression( None, 'x', None, None ) ]
  assert result.FuncName is None
  assert result.Operators == []


def test_leading_plus_raises() -> None:
  ''''+x' is rejected since a leading '+' is not valid syntax.'''
  with pytest.raises( ValueError ):
    ExpressionParser( "+x" )
