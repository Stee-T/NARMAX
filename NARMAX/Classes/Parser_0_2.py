import re
from dataclasses import dataclass

from typing import Optional

####################################################################################################################################################################################
#####                                                                                 STRUCTS                                                                                  #####
####################################################################################################################################################################################

################################################################################### Sub Expression #################################################################################
@dataclass
class SubExpression:
  Coeff: Optional[ float ] = None # multiplicative coefficient, if VarName is None then Subexpression is a constant
  VarName: Optional[ str ] = None
  Lag: Optional[ int ] = None
  Exponent: Optional[ float ] = None

  def __str__( self ) -> str: # Allows to print the object
    Out: str = "" # Everything is optional, so check if initialized
    if ( self.VarName is not None ): Out = self.VarName
    if ( self.Coeff is not None ):   Out = f"{ self.Coeff }*" + Out # prepend
    if ( self.VarName is None ):     Out = Out[:-1] # remove the '*' for scalars

    if ( self.VarName is not None ): # only add lag info for non-constants
      if ( self.Lag is None ):        Out += f"[k]" # append
      else:                           Out += f"[k{ self.Lag }]" # append

    if ( self.Exponent is not None ): Out += f"**{ self.Exponent }" # append
    return ( Out )
  
  def __eq__( self, other: object ) -> bool: # Comparison operator
    return ( self.__dict__ == other.__dict__ )


############################################################################## Regressor in String form ############################################################################
@dataclass
class ParsedReg:
  FuncName: Optional[ str ]
  SubExpressions: list[ SubExpression ] # List of 'SubExpression' variables
  Operators: list[ str ] # List of strings

  def __str__( self ) -> str: # Allows to print the object
    sub_expressions: str = ', '.join( [ str( se ) for se in self.SubExpressions ] )
    operators: str = ', '.join( self.Operators )
    return ( f"FuncName: { self.FuncName }\nSubExpressions: [{ sub_expressions }]\nOperators: [{ operators }]\n" )
  
  def __eq__( self, other: object ) -> bool: # Comparison operator
    return ( self.__dict__ == other.__dict__ )


####################################################################################################################################################################################
#####                                                                                  PARSER                                                                                  #####
####################################################################################################################################################################################

# ############################################################################### Expression Cleaner ###############################################################################
def CleanExpression( InputExpr: str ) -> tuple[ Optional[ str ], str ]:
  if ( not isinstance( InputExpr, str ) ): raise ValueError( f"Passed Expression '{ InputExpr }' is not a string" )

  # check for fractional lags and throw
  Lags = re.findall( r'\[(\w+)([+\-]\d+(\.\d+)?)\]', InputExpr )
  if ( len( Lags ) > 0 ):
    for lag in Lags:
      if ( lag[2] != '' ): raise ValueError( f"Expression: '{ InputExpr }' contains fractional lags, which is not supported as array index" )
  
  Expr: str = InputExpr.replace( '**', '^' ) # flatten potential python powers to normal powers

  # Handle missing spaces, excludes +,-,^ on purpose since this ruins the lags and exponents analysis
  for op in [ '*', '/' ]: Expr = Expr.replace( op, ' ' + op + ' ' )
  Expr = Expr.replace( '~ / ', '~/' ); Expr = Expr.replace( '1 / ', '1/' ) # repair two exceptions of the above space
  Expr = Expr.replace( '(', '( ' ); Expr = Expr.replace( ')', ' )' ) # Add spaces around parenthesis
  Expr = ' '.join( Expr.split() ) # collapse an arbitrary amount of spaces to a single one (the above might have created double spaces)

  # Merge subtractions into the coefficient
  Expr = re.sub( r'- (?=\d)', "+ -", Expr ) # replace stand-alone minus before digits
  Expr = re.sub( r'- (?=[a-zA-Z_])', '+ -1*', Expr ) # replace stand-alone minus before letters/underscore
  Expr = re.sub( r'-(?=[a-zA-Z_])', '-1*', Expr ) # replace not-stand-alone minus before letters/underscore

  # Merge multiplications into the Expression by deleting the spaces between digits and letters
  if ( Expr[0].isdigit() or ( Expr[0] == '-' ) ): Expr = ' ' + Expr # Space before multiplicative coefficients is required for the below coeff merging
  Expr = re.sub( r' (-?[\d.]+)\s*\* ([a-zA-Z_])', r' \1*\2', Expr ) # requires a space before the number to avoid matches like x^2 * y
  if ( Expr[0] == ' ' ): Expr = Expr[1:] # Remove the space (generated by the above, since otherwise eliminated by split)

  Parts: list[ str ] = Expr.split( '(' ) # Splitting the expression into function and its arguments → assume no parentheses

  if ( len( Parts ) == 1 ): # Split contains a single segment → no split performed = no function
    function_name: Optional[ str ] = None
    args = Parts[0]
  
  elif ( len( Parts ) == 2 ): # 'func' + '(' + 'args' + ')' case
    function_name: Optional[ str ] = Parts[0]
    args: str = Parts[1][:-1] # Removing the closing parenthesis

  else:  # anything else lol
    raise ValueError( f"Expression: '{ InputExpr }' contains multiple parentheses, which is currently not supported" )

  return ( function_name, args )


################################################################################# Expression Parser ################################################################################
def ExpressionParser( InputExpr: str ) -> ParsedReg:
  """Parses an expression and returns thecorresponding RegressorStr struct function name and a list of subexpressions"""
  function_name, args = CleanExpression( InputExpr ) # Pre-Process for clean parsing

  Subexpressions: list[ SubExpression ] = []
  Operators: list[ str ] = []

  for arg in args.split(): # Requires that all arguments be separated by spaces
    # every 2nd element in arg should be an operator but better check
    
    if ( arg in [ '+', '-', '*', '/', '^' ] ): # Not expecting stand-alone minuses, but who knows
      Operators.append( arg ) 
    
    else: # Subexpressions
      match = re.match( r'(-?[\d.]+)?\*?(\w+)?(?:\[(\w+)([+-]\d+)?\])?\^?([\d.]+)?', arg ) # Regexp for coefficient*variable[lag-lag_value]^exponent, all are optional
      if ( match ):
        coefficient = None if match.group(1) is None else float( match.group(1) )
        variable = match.group(2) # no processing since string
        # lagName = match.group(3) if match.group(3) else 'k' # Default lag variable, not of interest to us
        lag_value = None if match.group(4) is None else int( match.group(4) ) # float allows deciamls without throwing, int cuts it off
        exponent  = None if match.group(5) is None else float( match.group(5) )

        if ( variable is None ): # Must be a constant: Stored in the coefficient
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
  Regressor = ExpressionParser( InputExpr )
  print( "Parsing: ", InputExpr )
  print( "Function Name: ", Regressor.FuncName )
  print( "\nSubexpressions: " )

  RegSE = Regressor.SubExpressions # alias for brievty
  for pos in range( len( Regressor.SubExpressions ) ):
    print( f"\nCoefficient: { RegSE[pos].Coeff }\nVariable: { RegSE[pos].VarName }\nLag: { RegSE[pos].Lag }\nExponent: { RegSE[pos].Exponent }\n" )
    
    if ( pos != len( RegSE ) -1 ): # There is one less operator than subexpression (see: x + y - z)
      print ( "Operator:", Regressor.Operators[pos] )


####################################################################################################################################################################################
#####                                                                                UNIT TESTS                                                                                #####
####################################################################################################################################################################################

if ( __name__ == '__main__' ):

  Expressions = [
                 "sin(0.957*x[k-1]^3)", # function with coefficient * argument ^Exponent
                 "0.6x1[k+2]^11", # no function, no multiplication sign, number in varname, positive lag, double digit exponent
                 "-0.6x9^2", # no function, no multiplication sign, no time lag
                 "-0.5x[k-11] ^ 3*y[k-1]", # double digit lag, exponent with another expression
                 "exp(0.8*x_2[k-4] - 0.6*x_1[k]^2 * y[k-2])", # function, multiple arguments and coefficients, underscores in variable names
                 "-y[k-2] - 0.6x1[k-2]^3", # Minus without coefficient, minus not part of coefficient
                 "sin( 0.8 * x1[k-1]^3 - 0.2 )", # Scalar argument, random spaces (one of each making 0.8 a stanadlone Subexpression)
                 "~/(x1[k-1] + 0.8* x2[k-4])", # Denominator terms: ~/ Expr, unnecessary space between coeff and expression
                 "0.9*x1[k-1]^3 / 0.6*x1[k-2]^3", # Fractions
                 "0.9*x1^3 / 0.679 * x1[k-2]^3.5", # Fractions, no lag, unnecessary spaces (see comment below on how spaces imply parenthesis)
                 "~/abs(x1[k-1] + 0.8*x2[k-4])", # Parenthesis + non-lin denominator term
                 "1/abs(0.2*x[k] + 0.8*y2[j-4])", # Parenthesis + actual fraction with non-lin, other lag variable

                 # Weird but user's problem
                 "32*sin(0.9*x1[k-1]^3 + 0.6x1[k-2]^3)", # 32* is part of the func name, lol

                #  "x1[k-1.5]^3", # decimal lag correctly raises an error

                 # Not supported yet (essentially anything with omre than one prenthesis)
                 # "exp(0.8*x2[k-4] - 0.6*(x1[k]^2 + y[k-2]))", # Parenthesis
                 #  "sin( cos(0.9*x1[k-1]^3) + 0.6x1[k-2]^3)", # Parentheses + functions
                #  "sin(0.957*x1[k - 1]^3)", # random spaces inside the braces
  ]

  ParsedExpressions = [
    ParsedReg( 'sin',   [ SubExpression( 0.957, 'x', -1, 3.0 ) ], [] ),
    ParsedReg( None,    [ SubExpression( 0.6, 'x1', 2, 11.0 ) ], [] ),
    ParsedReg( None,    [ SubExpression( -0.6, 'x9', None, 2.0 ) ], [] ),
    ParsedReg( None,    [ SubExpression( -0.5, 'x', -11, None ), SubExpression( 3.0, 'y', -1, None ) ], ['^'] ),
    ParsedReg( 'exp',   [ SubExpression( 0.8, 'x_2', -4, None ), SubExpression( -0.6, 'x_1', None, 2.0 ), SubExpression( None, 'y', -2, None ) ], ['+', '*'] ),
    ParsedReg( None,    [ SubExpression( -1.0, 'y', -2, None ),  SubExpression( -0.6, 'x1', -2, 3.0 ) ], ['+'] ),
    ParsedReg( 'sin',   [ SubExpression( 0.8, 'x1', -1, 3.0 ),   SubExpression( -0.2, None, None, None ) ], ['+'] ),
    ParsedReg( '~/',    [ SubExpression( None, 'x1', -1, None ), SubExpression( 0.8, 'x2', -4, None ) ], ['+'] ),
    ParsedReg( None,    [ SubExpression( 0.9, 'x1', -1, 3.0 ),   SubExpression( 0.6, 'x1', -2, 3.0 ) ], ['/'] ),
    ParsedReg( None,    [ SubExpression( 0.9, 'x1', None, 3.0 ), SubExpression( 0.679, 'x1', -2, 3.5 ) ], ['/'] ),
    ParsedReg( '~/abs', [ SubExpression( None, 'x1', -1, None ), SubExpression( 0.8, 'x2', -4, None ) ], ['+'] ),
    ParsedReg( '1/abs', [ SubExpression( 0.2, 'x', None, None ), SubExpression( 0.8, 'y2', -4, None ) ], ['+'] ),
    ParsedReg( '32*sin',[ SubExpression( 0.9, 'x1', -1, 3.0 ),   SubExpression( 0.6, 'x1', -2, 3.0 ) ], ['+'] ),
  ]

  DebugExpression( Expressions[ 0 ] ) # just to test that function

  Fail: bool = False
  for test in range( len( Expressions ) ):
    if ( ExpressionParser( Expressions[ test ] ) != ParsedExpressions[ test ] ):
      Fail = True; print( 'ERROR: ' + Expressions[ test ] )
  if ( not Fail ): print( '\nAll tests passed!\n' )


# Note: spaces are important in the expressions and act as braces:
# - a/b * c is interpreted as (a/b) * c
# - a / b*c is interpreted as a / (b * c)

# float coefficients are merged with the term following them. This doesn't matter in most cases:
# - Expr1 * float * Expr2 is interpreted as Expr1 * (float * Expr2)
# But it can be weird (but not technically wrong) for divisions:
# - Expr1 / float * Expr2 is interpreted as Expr1 / (float * Expr2)
    
# TODO: 
# All above expressions are supported. This is missing:
# - internal parenthesis
# - exponentiations of functions
# - multiplicative coefficients of functions