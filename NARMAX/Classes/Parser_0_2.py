import re
from dataclasses import dataclass

from typing import Any, Optional, override

####################################################################################################################################################################################
#####                                                                                 STRUCTS                                                                                  #####
####################################################################################################################################################################################

################################################################################### Sub Expression #################################################################################
@dataclass
class SubExpression:
  '''Represents a single sub-expression with an optional coefficient, variable name, lag, and exponent.'''
  Coeff: Optional[ float ] = None # multiplicative coefficient, if VarName is None then Subexpression is a constant
  VarName: Optional[ str ] = None
  Lag: Optional[ int ] = None
  Exponent: Optional[ float ] = None

  @override
  def __str__( self ) -> str:
    '''Return a string representation of the SubExpression.'''
    Out: str = "" # Everything is optional, so check if initialized
    if ( self.VarName is not None ): Out = self.VarName
    if ( self.Coeff is not None ):   Out = f"{ self.Coeff }*" + Out # prepend
    if ( self.VarName is None ):     Out = Out[ : -1 ] # remove the '*' for scalars

    if ( self.VarName is not None ): # only add lag info for non-constants
      if ( self.Lag is None ):        Out += f"[k]" # append
      else:                           Out += f"[k{ self.Lag:+}]" # append and always include sign

    if ( self.Exponent is not None ): Out += f"**{ self.Exponent }" # append
    return ( Out )


############################################################################## Regressor in String form ############################################################################
@dataclass
class ParsedReg:
  '''Represents a parsed regressor with a function name, sub-expressions, and operators.'''
  FuncName: Optional[ str ]
  SubExpressions: list[ SubExpression ] # List of 'SubExpression' variables
  Operators: list[ str ] # List of strings

  @override
  def __str__( self ) -> str:
    '''Return a string representation of the ParsedReg.'''
    sub_expressions: str = ', '.join( [ str( se ) for se in self.SubExpressions ] )
    operators: str = ', '.join( self.Operators )
    return ( f"FuncName: { self.FuncName }\nSubExpressions: [{ sub_expressions }]\nOperators: [{ operators }]\n" )


####################################################################################################################################################################################
#####                                                                                  PARSER                                                                                  #####
####################################################################################################################################################################################

# ############################################################################### Expression Cleaner ###############################################################################

def CleanExpression( InputExpr: str ) -> tuple[ Optional[ str ], str ]:
  '''Clean and pre-process an expression string for parsing.

  Args:
      InputExpr: The expression string to clean.

  Returns:
      A tuple of (function_name, args) where function_name may be None.

  Raises:
      ValueError: If the expression contains reserved sequences, fractional lags,
          multiple parentheses, or function names with spaces.
  '''
  if ( not isinstance( InputExpr, str ) ): raise ValueError( f"Passed Expression '{ InputExpr }' is not a string" )

  # 0. Check for reserved internal sequence
  if ( "__LAG" in InputExpr ): raise ValueError( "The character sequence '__LAG' is reserved for internal processing and may not appear in expressions." )

  # 0.1 Remove spaces inside brackets
  InputExpr = re.sub( r'\[\s*(.*?)\s*\]', r'[\1]', InputExpr )

  # --- fractional lag check (improved) ---
  if ( re.findall( r'\[\w+[+-](\d+\.\d*|\d*\.\d+)\]', InputExpr ) ):
    raise ValueError( f"Expression: '{ InputExpr }' contains fractional lags, which is not supported as array index" )

  Expr: str = InputExpr.replace( '**', '^' )

  # 1. Protect lag brackets so we can safely add spaces around +/-
  lag_pattern = re.compile( r'\[[^\]]+\]' )
  lags_found = lag_pattern.findall( Expr )
  for i, lag in enumerate( lags_found ): Expr = Expr.replace( lag, f'__LAG{ i }__', 1 )

  # 2. Add spaces around binary + and - (only when between word chars, ] or ) )
  Expr = re.sub( r'(?<=[\w\)\]]) *([+\-]) *(?=[\w\(])', r' \1 ', Expr )

  # 3. Restore the original lag brackets
  for i, lag in enumerate( lags_found ): Expr = Expr.replace( f'__LAG{ i }__', lag, 1 )

  # 4. Handle missing spaces around * and /
  for op in [ '*', '/' ]: Expr = Expr.replace( op, ' ' + op + ' ' )

  # 5. Robust repair of ~/ and 1/ operators (with or without a function name)
  Expr = re.sub( r'~\s*/\s*(\w+)\s*\(', r'~/\1(', Expr )
  Expr = re.sub( r'~\s*/\s*(?=\()', r'~/', Expr )
  Expr = re.sub( r'1\s*/\s*(\w+)\s*\(', r'1/\1(', Expr )
  Expr = re.sub( r'1\s*/\s*(?=\()', r'1/', Expr )

  # 6. Add spaces around parentheses
  Expr = Expr.replace( '(', '( ' ).replace( ')', ' )' )

  # 7. Collapse multiple spaces
  Expr = ' '.join( Expr.split() )

  # 8. Merge subtractions into the coefficient (original behaviour)
  Expr = re.sub( r'- (?=\d)', "+ -", Expr )
  Expr = re.sub( r'- (?=[a-zA-Z_])', '+ -1*', Expr )
  Expr = re.sub( r'-(?=[a-zA-Z_])', '-1*', Expr )

  # 9. Merge multiplications into the Expression by deleting the spaces between digits and letters
  if ( Expr[ 0 ].isdigit() or ( Expr[ 0 ] == '-' ) ): Expr = ' ' + Expr
  Expr = re.sub( r' (-?[\d.]+)\s*\* ([a-zA-Z_])', r' \1*\2', Expr )
  if ( Expr[ 0 ] == ' ' ): Expr = Expr[ 1 : ]

  # 10. Split function name & arguments
  Parts: list[ str ] = Expr.split( '(' )

  if ( len( Parts ) == 1 ):
    function_name: Optional[ str ] = None
    args: str = Parts[ 0 ]
  elif ( len( Parts ) == 2 ):
    function_name = Parts[ 0 ].strip()
    # Reject function names that contain whitespace
    if ( function_name and ( re.search( r'\s', function_name ) ) ):
      raise ValueError( f"Expression: '{ InputExpr }' contains a function name with spaces: '{ function_name }'" )
    args = Parts[ 1 ].rstrip().removesuffix( ')' )
  else:
    raise ValueError( f"Expression: '{ InputExpr }' contains multiple parentheses, which is currently not supported" )

  return ( function_name, args )

################################################################################# Expression Parser ################################################################################
def ExpressionParser( InputExpr: str ) -> ParsedReg:
  '''Parses an expression and returns the corresponding ParsedReg with function name, sub-expressions, and operators.'''
  function_name, args = CleanExpression( InputExpr ) # Pre-Process for clean parsing

  Subexpressions: list[ SubExpression ] = []
  Operators: list[ str ] = []

  for arg in args.split(): # Requires that all arguments be separated by spaces
    # every 2nd element in arg should be an operator but better check

    if ( arg in [ '+', '-', '*', '/', '^' ] ): Operators.append( arg ) # Not expecting stand-alone minuses, but who knows

    else: # Subexpressions
      # Regexp for coefficient*variable[lag-lag_value]^exponent
      # Now allows a lone '-' as the coefficient (meaning -1)
      match: Optional[ re.Match[ str ] ] = re.fullmatch( r'(-?[\d.]*)\*?(\w+)?(?:\[(\w+)([+-]\d+)?\])?\^?([\d.]+)?', arg )
      if ( match ):
        CoeffRaw = match.group( 1 )
        if ( CoeffRaw is None or CoeffRaw == '' ): coefficient = None
        elif ( CoeffRaw == '-' ):                  coefficient = -1.0
        else:
          try:
            coefficient = float( CoeffRaw )
          except ValueError:
            raise ValueError( f"Invalid numeric coefficient '{ CoeffRaw }' in sub-expression '{ arg }'" )

        variable: Optional[ str ] = match.group( 2 )
        lag_value: Optional[ int ] = None if match.group( 4 ) is None else int( match.group( 4 ) )
        exponent: Optional[ float ] = None if match.group( 5 ) is None else float( match.group( 5 ) )

        if ( variable is None ):
          if ( coefficient is None ):   raise ValueError( f"Invalid Expression '{ arg }': Not a variable nor a constant" )
          if ( lag_value is not None ): raise ValueError( f"Pointless Expression '{ arg }': Lag without variable" )
          if ( exponent is not None ):  raise ValueError( f"Pointless Expression '{ arg }': Exponent without variable" )

        Subexpressions.append( SubExpression( coefficient, variable, lag_value, exponent ) )
      else: raise ValueError( f"Expression: '{ arg }' was not recognized as a valid expression" )

  if ( len( Operators ) != len( Subexpressions ) - 1 ):
    raise ValueError( f"Mismatch between the number of expressions and operators in '{ InputExpr }'" )

  return ( ParsedReg( function_name, Subexpressions, Operators ) )


################################################################################ Expression Debugger ###############################################################################
def DebugExpression( InputExpr: str ) -> None:
  '''Parse and print the structure of an expression for debugging.'''
  Regressor = ExpressionParser( InputExpr )
  print( "Parsing: ", InputExpr )
  print( "Function Name: ", Regressor.FuncName )
  print( "\nSubexpressions: " )

  RegSE: list[ SubExpression ] = Regressor.SubExpressions # alias for brievty
  for pos in range( len( Regressor.SubExpressions ) ):
    print( f"\nCoefficient: { RegSE[ pos ].Coeff }\nVariable: { RegSE[ pos ].VarName }\nLag: { RegSE[ pos ].Lag }\nExponent: { RegSE[ pos ].Exponent }\n" )

    if ( pos != len( RegSE ) - 1 ): print ( "Operator:", Regressor.Operators[ pos ] ) # There is one less operator than subexpression (see: x + y - z)

####################################################################################################################################################################################

# Note: spaces are important in the expressions and act as parenths:
# - a/b * c is interpreted as (a/b) * c
# - a / b*c is interpreted as a / (b * c)

# float coefficients are merged with the term following them. This doesn't matter in most cases:
# - Expr1 * float * Expr2 is interpreted as Expr1 * (float * Expr2)
# But it can be weird (but not technically wrong) for divisions:
# - Expr1 / float * Expr2 is interpreted as Expr1 / (float * Expr2)
