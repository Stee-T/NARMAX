import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, override

####################################################################################################################################################################################
#####                                                                                 STRUCTS                                                                                  #####
####################################################################################################################################################################################

############################################################################## ExprNode (unifies SubExpression + ParsedReg) ##########################################################
@dataclass
class ExprNode:
  '''Unified AST node representing either a leaf (variable/constant) or a branch (grouped expression/function).'''
  Coeff: Optional[ float ] = None # multiplicative coefficient, if VarName is None then node is a constant (leaf)
  VarName: Optional[ str ] = None
  Lag: Optional[ int ] = None
  Exponent: Optional[ float ] = None

  FuncName: Optional[ str ] = None # function name (e.g. 'sin', '~/', '1/'); None for leaf / group-only nodes
  SubExpressions: list[ 'ExprNode' ] = field( default_factory = list ) # child nodes for grouped/function expressions
  Operators: list[ str ] = field( default_factory = list ) # operators between sub-expressions

  @property
  def is_leaf( self ) -> bool: return not self.SubExpressions and ( self.FuncName is None )

  @override
  def __str__( self ) -> str:
    if ( self.is_leaf ):
      out = ""
      if ( self.VarName is not None ): out = self.VarName
      if ( self.Coeff is not None ): out = f"{ self.Coeff }*" + out if self.VarName else f"{ self.Coeff }"
      if ( self.VarName is not None ): out += f"[k]" if self.Lag is None else f"[k{ self.Lag:+}]"
      if ( self.Exponent is not None ): out += f"**{ self.Exponent }"
      return out # R[1/2]
    else:
      sub = ", ".join( str( s ) for s in self.SubExpressions )
      ops = ", ".join( self.Operators )
      func = f"{ self.FuncName }(" if self.FuncName else "("
      coeff = f"{ self.Coeff }*" if self.Coeff is not None else ""
      return f"{ coeff }{ func }{ sub } ops:[{ ops }])" # R[2/2]

####################################################################################################################################################################################
#####                                                                                  PARSER                                                                                  #####
####################################################################################################################################################################################

class TokenType( Enum ):
  NUMBER = auto()
  NAME = auto()
  OP = auto()
  LPAREN = auto()
  RPAREN = auto()
  LBRACK = auto()
  RBRACK = auto()
  RATIONAL = auto()

TOKEN_SPEC: list[ tuple[ str, str ] ] = [
  ( 'RATIONAL', r'(?:~/|1/)(?=\(|[a-zA-Z_])' ),
  ( 'NUMBER', r'\d+\.\d*|\.\d+|\d+(?:[eE][+-]?\d+)?' ),
  ( 'NAME', r'[a-zA-Z_]\w*' ),
  ( 'OP', r'[+\-*/^]' ),
  ( 'LPAREN', r'\(' ),
  ( 'RPAREN', r'\)' ),
  ( 'LBRACK', r'\[' ),
  ( 'RBRACK', r'\]' ),
  ( 'SKIP', r'[ \t]+' ),
  ( 'MISMATCH', r'.' ),
]
TokRegex: str = '|'.join( '(?P<%s>%s)' % pair for pair in TOKEN_SPEC )

def CleanExpression( InputExpr: str ) -> str:
  '''Pre-processes string: removes spaces in brackets, fixes fractional lags, and repairs rational prefixes.

  Args:
      InputExpr: The expression string to clean.

  Returns:
      A single cleaned expression string. Function names and arguments are no longer split here,
      as the recursive descent parser handles arbitrary depth parentheses.

  Raises:
      ValueError: If the expression contains reserved sequences, fractional lags,
          malformed numbers, or function names with spaces.
  '''
  if ( not isinstance( InputExpr, str ) ): raise ValueError( f"Passed Expression '{ InputExpr }' is not a string" )
  if ( "__LAG" in InputExpr ): raise ValueError( "The character sequence '__LAG' is reserved for internal processing." )

  Expr: str = re.sub( r'\[\s*(.*?)\s*\]', r'[\1]', InputExpr )
  if ( re.findall( r'\[\w+[+-](\d+\.\d*|\d*\.\d+)\]', Expr ) ):
    raise ValueError( f"Expression: '{ InputExpr }' contains fractional lags, which is not supported" )

  Expr = Expr.replace( '**', '^' )

  # Reject malformed numbers (e.g. 1.2.3)
  if ( re.search( r'\d+\.\d+\.\d+', Expr ) ): raise ValueError( "Invalid numeric coefficient" )

  # Reject spaces inside function names (e.g. "my func(x)")
  if ( re.search( r'[a-zA-Z_]\w*\s+[a-zA-Z_]\w*\s*\(', Expr ) ):
    raise ValueError( "Expression contains a function name with spaces" )

  # Repair rational prefixes with spaces
  Expr = re.sub( r'~\s*/\s*(\w+)\s*\(', r'~/\1(', Expr )
  Expr = re.sub( r'~\s*/\s*(?=\()', r'~/', Expr )
  Expr = re.sub( r'1\s*/\s*(\w+)\s*\(', r'1/\1(', Expr )
  Expr = re.sub( r'1\s*/\s*(?=\()', r'1/', Expr )

  return Expr


def Tokenize( expr: str ) -> list[ tuple[ TokenType, str ] ]:
  Tokens: list[ tuple[ TokenType, str ] ] = []
  for mo in re.finditer( TokRegex, expr ):
    kind: str = mo.lastgroup
    value: str = mo.group()
    if ( kind == 'SKIP' ): continue
    elif ( kind == 'MISMATCH' ): raise ValueError( f"Unexpected character: { value }" )
    else: Tokens.append( ( TokenType[ kind ], value ) )
  return Tokens


def InsertImplicitMultiplication( Tokens: list[ tuple[ TokenType, str ] ] ) -> list[ tuple[ TokenType, str ] ]:
  Out: list[ tuple[ TokenType, str ] ] = []
  for tokenIdx in range( len( Tokens ) ):
    Out.append( Tokens[ tokenIdx ] )
    if ( tokenIdx < len( Tokens ) - 1 ):
      curr: TokenType = Tokens[ tokenIdx ][ 0 ]
      nxt: TokenType = Tokens[ tokenIdx + 1 ][ 0 ]
      mul: bool = False
      if ( ( curr == TokenType.NUMBER ) and ( nxt in ( TokenType.NAME, TokenType.LPAREN, TokenType.RATIONAL ) ) ): mul = True
      elif ( ( curr == TokenType.RPAREN ) and ( nxt in ( TokenType.NUMBER, TokenType.NAME, TokenType.LPAREN, TokenType.RATIONAL ) ) ): mul = True
      elif ( ( curr == TokenType.RBRACK ) and ( nxt in ( TokenType.NUMBER, TokenType.NAME, TokenType.LPAREN, TokenType.RATIONAL ) ) ): mul = True
      if ( mul ): Out.append( ( TokenType.OP, '*' ) )
  return Out


class Parser:
  Tokens: list[ tuple[ TokenType, str ] ]
  Pos: int

  def __init__( self, Tokens: list[ tuple[ TokenType, str ] ] ) -> None:
    self.Tokens = Tokens
    self.Pos = 0

  def Peek( self ) -> Optional[ tuple[ TokenType, str ] ]:
    return self.Tokens[ self.Pos ] if ( self.Pos < len( self.Tokens ) ) else None

  def Consume( self, expected_type: Optional[ TokenType ] = None, expected_val: Optional[ str ] = None ) -> tuple[ TokenType, str ]:
    tok: Optional[ tuple[ TokenType, str ] ] = self.Peek()
    if ( tok is None ): raise ValueError( "Unexpected end of expression" )
    if ( expected_type and ( tok[ 0 ] != expected_type ) ):
      raise ValueError( f"Expected { expected_type.name }, got { tok[ 0 ].name } ('{ tok[ 1 ] }')" )
    if ( expected_val and ( tok[ 1 ] != expected_val ) ):
      raise ValueError( f"Expected '{ expected_val }', got '{ tok[ 1 ] }'" )
    self.Pos += 1
    return tok

  def Parse( self ) -> ExprNode:
    if ( not self.Tokens ): raise ValueError( "Empty expression" )
    node: ExprNode = self.ParseExpr()
    if ( self.Pos < len( self.Tokens ) ): raise ValueError( f"Unexpected token at end: { self.Peek()[ 1 ] }" )
    return node

  def ParseExpr( self ) -> ExprNode:
    Nodes: list[ ExprNode ] = []
    Ops: list[ str ] = []
    Sign: float = 1.0
    if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) and ( self.Peek()[ 1 ] in ( '+', '-' ) ) ):
      if ( self.Consume( TokenType.OP )[ 1 ] == '-' ): Sign = -1.0

    if ( ( not self.Peek() ) or ( self.Peek()[ 0 ] in ( TokenType.RPAREN, TokenType.RBRACK ) ) ):
      raise ValueError( "Invalid expression structure" )

    Nodes.append( self.parse_atom( Sign ) )

    while ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) ):
      op: str = self.Consume( TokenType.OP )[ 1 ]
      NextSign: float = 1.0
      if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) and ( self.Peek()[ 1 ] in ( '+', '-' ) ) ):
        if ( self.Consume( TokenType.OP )[ 1 ] == '-' ): NextSign = -1.0

      if ( op == '-' ):
        Ops.append( '+' )
        NextSign *= -1.0
      else: Ops.append( op )

      Nodes.append( self.parse_atom( NextSign ) )

    return ExprNode( SubExpressions = Nodes, Operators = Ops )

  def parse_atom( self, sign: float = 1.0 ) -> ExprNode:
    tok: Optional[ tuple[ TokenType, str ] ] = self.Peek()
    if ( tok is None ): raise ValueError( "Unexpected end of expression" )

    if ( tok[ 0 ] == TokenType.RATIONAL ):
      Prefix: str = self.Consume( TokenType.RATIONAL )[ 1 ]
      next_tok: Optional[ tuple[ TokenType, str ] ] = self.Peek()
      if ( next_tok and ( next_tok[ 0 ] == TokenType.NAME ) ):
        name: str = Prefix + self.Consume( TokenType.NAME )[ 1 ]
        self.Consume( TokenType.LPAREN )
        inner: ExprNode = self.ParseExpr()
        self.Consume( TokenType.RPAREN )
        inner.FuncName = name
        if ( sign != 1.0 ): inner.Coeff = ( inner.Coeff or 1.0 ) * sign
        return inner # R[1/6]
      elif ( next_tok and ( next_tok[ 0 ] == TokenType.LPAREN ) ):
        name = Prefix
        self.Consume( TokenType.LPAREN )
        inner = self.ParseExpr()
        self.Consume( TokenType.RPAREN )
        inner.FuncName = name
        if ( sign != 1.0 ): inner.Coeff = ( inner.Coeff or 1.0 ) * sign
        return inner # R[2/6]
      else: raise ValueError( f"Rational prefix '{ Prefix }' must be followed by a function name or parenthesis" )

    if ( tok[ 0 ] == TokenType.NUMBER ):
      self.Consume()
      return ExprNode( Coeff = float( tok[ 1 ] ) * sign ) # R[3/6]

    elif ( tok[ 0 ] == TokenType.NAME ):
      name = tok[ 1 ]
      if ( ( self.Pos + 1 < len( self.Tokens ) ) and ( self.Tokens[ self.Pos + 1 ][ 0 ] == TokenType.LPAREN ) ):
        self.Consume( TokenType.NAME )
        self.Consume( TokenType.LPAREN )
        inner = self.ParseExpr()
        self.Consume( TokenType.RPAREN )
        inner.FuncName = name
        if ( sign != 1.0 ): inner.Coeff = ( inner.Coeff or 1.0 ) * sign
        return inner # R[4/6]
      else:
        self.Consume( TokenType.NAME )
        Lag: Optional[ int ] = None
        exponent: Optional[ float ] = None
        if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.LBRACK ) ):
          self.Consume( TokenType.LBRACK )
          self.Consume( TokenType.NAME )
          if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) and ( self.Peek()[ 1 ] in ( '+', '-' ) ) ):
            op = self.Consume( TokenType.OP )[ 1 ]
            num = self.Consume( TokenType.NUMBER )[ 1 ]
            try: Lag = int( op + num )
            except ValueError: raise ValueError( "Expression contains fractional lags, which is not supported" )
          self.Consume( TokenType.RBRACK )

        if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) and ( self.Peek()[ 1 ] == '^' ) ):
          CanConsume: bool = False
          if ( self.Pos + 1 < len( self.Tokens ) ):
            nxt: tuple[ TokenType, str ] = self.Tokens[ self.Pos + 1 ]
            if ( nxt[ 0 ] == TokenType.NUMBER ): CanConsume = True
            elif ( ( nxt[ 0 ] == TokenType.OP ) and ( nxt[ 1 ] in ( '+', '-' ) ) and ( self.Pos + 2 < len( self.Tokens ) ) ):
              nxt2: tuple[ TokenType, str ] = self.Tokens[ self.Pos + 2 ]
              if ( nxt2[ 0 ] == TokenType.NUMBER ): CanConsume = True
          if ( CanConsume ):
            self.Consume( TokenType.OP, '^' )
            ExpSign: float = 1.0
            if ( ( self.Peek() ) and ( self.Peek()[ 0 ] == TokenType.OP ) and ( self.Peek()[ 1 ] in ( '+', '-' ) ) ):
              if ( self.Consume( TokenType.OP )[ 1 ] == '-' ): ExpSign = -1.0
            exponent = float( self.Consume( TokenType.NUMBER )[ 1 ] ) * ExpSign

        node: ExprNode = ExprNode( VarName = name, Lag = Lag, Exponent = exponent )
        if ( sign != 1.0 ): node.Coeff = sign
        return node # R[5/6]

    elif ( tok[ 0 ] == TokenType.LPAREN ):
      self.Consume( TokenType.LPAREN )
      inner = self.ParseExpr()
      self.Consume( TokenType.RPAREN )
      if ( sign != 1.0 ): inner.Coeff = ( inner.Coeff or 1.0 ) * sign
      return inner # R[6/6]

    raise ValueError( f"Unexpected token: { tok[ 1 ] }" )

################################################################################# Expression Parser ################################################################################
def ExpressionParser( InputExpr: str ) -> ExprNode:
  return Parser( InsertImplicitMultiplication( Tokenize( CleanExpression( InputExpr ) ) ) ).Parse()

################################################################################ Expression Debugger ###############################################################################
def DebugExpression( InputExpr: str ) -> None:
  '''Parse and print the structure of an expression for debugging.'''
  print( "Parsing: ", InputExpr )
  print( ExpressionParser( InputExpr ) )
