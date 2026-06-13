"""This submodule contains function and signal constructors for some common fitting types :
- `Lagger`: Creates delayed version of the signal. (x → x[k-j], j in some set )
- `Expander`: Creates monomial expansions of terms with products or powers of passed terms (x1, x2 → x1^2*x2)
- `Non-Linearizer`: apply a list of non-linearities (x→f(x)) and make them rational if desired (x → 1/f(x))
- `Booler`: Creates boolean logic combinations of binary regressors.
- `DerivativeAugment`: Augments the regressor matrix with smoothed derivatives via differentiated mollifiers.
"""

import numpy as np
import torch as tor
import itertools as it
import warnings
import matplotlib.pyplot as plt

from typing import Optional, Tuple, Sequence, Union, Callable
from numpy.typing import NDArray

from .. import HelperFuncs as HF # import parent module Helperfunctions

import sys
import os
sys.path.append( os.path.abspath( os.path.join( os.path.dirname( __file__ ), '..' ) ) ) # acess sibling folder
from Classes.NonLinearity import NonLinearity

#################################################################################### Lagger CTor ###################################################################################
def Lagger( Data: Sequence[ tor.Tensor ], Lags: Sequence[ Union[ int, Sequence[ int ] ] ], RegNames: Optional[ Union[ Sequence[ str ], NDArray[ np.str_ ] ] ] = None,
          ) -> Tuple[ Optional[ tor.Tensor ], tor.Tensor, NDArray[ np.str_ ] ]:
  '''Create delayed versions of input regressors.

  Each regressor in ``Data`` is shifted by the specified lags to produce a
  regressor matrix of lagged terms. If ``"y"`` is among the regressor names,
  its zero-lag column (``y[k]``) is split out and returned separately.

  Parameters
  ----------
  Data : Sequence[torch.Tensor]
      Sequence of (p,)-shaped tensors, one per regressor.
  Lags : Sequence[int or Sequence[int]]
      Per-regressor lag specification: an int gives lags 0..n, while a
      sequence of ints gives explicit lag values.
  RegNames : Sequence[str] or NDArray[np.str_], optional
      Names of the regressors. If None and ``len(Data) <= 3``, defaults
      to ``["x", "y", "e"]``.

  Returns
  -------
  y : torch.Tensor or None
      (p-q,)-shaped tensor of ``y[k]``, or None if no ``"y"`` regressor.
  RegMat : torch.Tensor
      (p-q, n_regs) tensor of lagged regressors.
  OutNames : NDArray[np.str_]
      Array of column names such as ``"x[k-1]"``.
  '''

  # --------------------------------------------------------------------------- A) # Bullshit prevention ---------------------------------------------------------------------------
  if ( len( Data ) == 0 ): raise ValueError( "Data must contain at least one regressor Matrix." )

  if ( len( Data ) != len( Lags ) ): raise AssertionError( "The numbers of regressors and lags don't correspond" )

  if ( RegNames is not None ):
    if ( len( Data ) != len( RegNames ) ): raise AssertionError( "The numbers of regressors and RegNames don't correspond" )

  for i in range( len( Data ) ):
    if ( not tor.isfinite( Data[ i ] ).all() ): raise AssertionError( f"Data[ { i } ] contains inf or nans. The to-be-fitted system is unstable in general or for this particular sequence. Try a new one" )

  for i in range( 1, len( Data ) ):
    if ( len( Data[ 0 ] ) != len( Data[ i ] ) ): raise AssertionError( "All regressors must have the same lenght" )

  for i in range( len( Lags ) ):
    if ( isinstance( Lags[ i ], int ) ):
      if ( Lags[ i ] < 0 ): raise AssertionError( f"Integer Lags elements must be >= 0, which is not the case for the { i }-th element" )

    elif ( isinstance( Lags[ i ], ( list, tuple ) ) ): # if sequence verify that all elements are integers
      if ( len( Lags[ i ] ) == 0 ): raise ValueError( f"Lags[ { i } ] must not be an empty sequence." )
      for j in range( len( Lags[ i ] ) ):
        if ( not isinstance( Lags[ i ][ j ], int ) ): raise AssertionError( f"All Lags-sublist elements must be integers, which is not the case for Lags[{ i }][{ j }]" )
        if ( Lags[ i ][ j ] < 0 ):                    raise AssertionError( f"All Lags-sublist elements must be integers >= 0, which is not the case for Lags[{ i }][{ j }]" )

    else:                                             raise AssertionError( f"All MaxLags elements must be integers or Lists, which is not the case for Lags[{ i }]" )

  # ------------------------------------------------------------------------------- B) Initialization ------------------------------------------------------------------------------
  if ( RegNames is None ):
    if ( len( Data ) <= 3 ): RegNames = [ "x", "y", "e" ][ : len( Data ) ]
    else:                    raise AssertionError( "No regressor names were passed and len( Data ) > 3. Thus x, y, e can't be assumed and RegNames must be passed" )

  q: int = 0 # Maximum lag to trim all regressors to
  for lag in Lags:
    if ( isinstance( lag, int ) ): q = max( q, lag )
    else:                          q = max( q, max( lag ) ) # sequence can contain whatever

  p: int = Data[ 0 ].view( -1, 1 ).shape[ 0 ] # number of samples in the regressors, all guaranteed to be the same
  if ( q >= p ): raise ValueError( f"At least one lag (q = { q }) is greater than the number of samples (p = { p }). Not enough Data to create desired lags" )

  RegMat: list[ tor.Tensor ] = [] # Regressor matrix big enough for all simple and optionally combined swung-in terms
  OutNames: list[ str ] = [] # Regressor RegNames

  # ------------------------------------------------------------------------- C) Delayed-copy constructors -------------------------------------------------------------------------
  for reg in range( len( Data ) ):

    if ( isinstance( Lags[ reg ], int ) ): LagList: list[ int ] = [ i for i in range( Lags[ reg ] + 1 ) ] # upperbound passed, so take all lags, +1 to include the passed Maxlag
    else:                                  LagList: list[ int ] = list( Lags[ reg ] ) # pre-determined list (copy to avoid mutating original if needed)

    if ( len( set( LagList ) ) != len( LagList ) ): raise ValueError( f"Duplicate lag values found in Lags[{ reg }] (regressor '{ RegNames[ reg ] }'). Each lag must be unique." )

    for lag in LagList:
      RegMat.append( Data[ reg ].view( -1 )[ q - lag : p - lag ] ) # flatten and take needed slice

      if ( lag == 0 ): OutNames.append( f"{ RegNames[ reg ] }[k]" ) # delayed name
      else:            OutNames.append( f"{ RegNames[ reg ] }[k-{ lag }]" ) # delayed name

  # ------------------------------------------------------------------------------- D) y[k] handling -------------------------------------------------------------------------------
  y: Optional[ tor.Tensor ] = None

  yPos: int = -1
  for i in range( len( RegNames ) ): # use for loop since RegNames is an arbitrary container and thus not guaranteed to have any particular member function
    if ( RegNames[ i ] == "y" ): yPos = i; break

  if ( yPos != -1 ): # a y was passed, so don't return the default None

    # If Lags[ yPos ] == int, then y[ k ] exists, since lags <= 0 are illegal, else it depends on Lags[ yPos ]'s content
    for i in range( len( OutNames ) ):
      if ( OutNames[ i ] == "y[k]" ): # see if y[ k ] exists and separate it
        y = RegMat[ i ].view( -1 )
        RegMat.pop( i )
        OutNames.pop( i )
        break

    if ( y is None ): y = Data[ yPos ].view( -1 )[ q : p ] # same as above but lag = 0

  # Avoid column_stack on empty list (only the case if RegMat containing only y[k] is passed, which is eliminated above)
  if ( not RegMat ): RegMatTensor = tor.empty( ( p - q, 0 ) )
  else:              RegMatTensor = tor.column_stack( RegMat )

  return ( y, RegMatTensor, np.array( OutNames, dtype = np.str_ ) )


# ############################################################################### Monomial Expansion ###############################################################################
def Expander( Data: tor.Tensor, RegNames: Union[ Sequence[ str ], NDArray[ np.str_ ] ], ExpansionOrder: int, IteractionOnly: bool = False
            ) -> ( tuple[ tor.Tensor, NDArray[ np.str_ ] ] ):
  '''Create monomial expansions of regressor terms.

  Generates all monomial combinations up to the given expansion order. With
  ``IteractionOnly=True``, only mixed-term products are produced (no self-powers).

  Parameters
  ----------
  Data : torch.Tensor
      (p, nC) tensor of regressors arranged columnwise.
  RegNames : Sequence[str] or NDArray[np.str_]
      (nC,) names of each input regressor column.
  ExpansionOrder : int
      Maximum monomial order (number of factors multiplied together).
  IteractionOnly : bool
      If True, only cross-terms (e.g. ``x[k] * x[k-1]``) are generated;
      powers of a single term (e.g. ``x[k]^2``) are omitted.

  Returns
  -------
  RegMat : torch.Tensor
      (p, nRegs) tensor of expanded monomial regressors.
  OutNames : NDArray[np.str_]
      (nRegs,) array of column names, e.g. ``"x[k]^2 * y[k-1]"``.
  '''

  if ( ( ExpansionOrder < 1 ) or ( not isinstance( ExpansionOrder, ( int, np.integer ) ) ) ): raise ValueError( "ExpansionOrder should be an int >= 1" )
  if ( ExpansionOrder == 1 ): return ( Data, np.array( RegNames ) ) # R[1/2] Nothing happens for 1-order polynomial expansion

  if ( Data.ndim != 2 ): raise ValueError( "Data should be 2-dimensional with regressors as columns" )
  if ( Data.shape[ 1 ] != len( RegNames ) ): raise ValueError( "Number of names does not match number of columns in the Data" )

  # --------------------------------------------------------------------------- A) Regressor Computation ---------------------------------------------------------------------------
  # Compute the total number of Regressors and pre-allocate memory
  if ( IteractionOnly ): nRegs: int = np.sum( [ HF.Combinations( Data.shape[ 1 ], o ) for o in range( 1, ExpansionOrder + 1 ) ] )
  else:                  nRegs: int = HF.Combinations( Data.shape[ 1 ] + ExpansionOrder, ExpansionOrder ) - 1

  RegMat: tor.Tensor = tor.full( ( Data.shape[ 0 ], nRegs ), np.nan )

  # degree 1 Expansion terms (no changes)
  RegMat[ :, : Data.shape[ 1 ] ] = Data
  index: list[ int ] = list( range( Data.shape[ 1 ] ) )
  currentCol: int = Data.shape[ 1 ]
  index.append( currentCol )

  for _ in range( 2, ExpansionOrder + 1 ): # loop over degree >= 2 terms, is skipped if ExpansionOrder = 1
    new_index: list[ int ] = []
    end: int = index[ -1 ]

    for feature_idx in range( Data.shape[ 1 ] ):
      start = index[ feature_idx ]
      new_index.append( currentCol )
      if ( IteractionOnly ): start += index[ feature_idx + 1 ] - index[ feature_idx ]
      next_col = currentCol + ( end - start ) # next column index being current + current lenght
      if ( next_col <= currentCol ): break # don't overshoot

      # RegMat[ :, start:end ] are terms of degree d - 1 that exclude feature feature_idx.
      RegMat[ :, currentCol : next_col ] = RegMat[ :, start : end ] * Data[ :, feature_idx : feature_idx + 1 ] # elementwise multiplication
      currentCol = next_col

    new_index.append( currentCol )
    index = new_index

  # --------------------------------------------------------------------------- B) RegNames construction ---------------------------------------------------------------------------
  OutNames: list[ Optional[ str ] ] = [ None ] * nRegs # Pre-allocate memory for speed, use a list, since numpy requires to know the maximum string length in advance

  Comb = it.combinations if IteractionOnly else it.combinations_with_replacement # chose correct combinations function
  iter = it.chain.from_iterable( Comb( range( Data.shape[ 1 ] ), i ) for i in range( 1, ExpansionOrder + 1 ) ) # no need for [] since itertools take variable number of arguments

  # tuples are sorted, being made of sorted containers, thus one can simply count the number of duplicate elements for the powers
  for counter, idx_tuple in enumerate( iter ): # iterate over all combinations of regressors
    Str = "" # reinitialize the string
    IndexSet, PowerSet = np.unique( idx_tuple, return_counts = True ) # remove duplicates and count them

    for idx in range( len( IndexSet ) ):
      Str += f"{ RegNames[ IndexSet[ idx ] ] } * " if ( PowerSet[ idx ] == 1 ) else f"{ RegNames[ IndexSet[ idx ] ] }^{ PowerSet[ idx ] } * "

    OutNames[ counter ] = Str[ : -3 ] # Cut last " * "

  return ( RegMat, np.array( OutNames ) ) # R[2/2]


# ########################################################################### Regressor Matrix Transform ###########################################################################
def NonLinearizer( y: Optional[ tor.Tensor ], Data: tor.Tensor, RegNames: Union[ Sequence[ str ], NDArray[ np.str_ ] ],
                   Functions: Sequence[ NonLinearity ], MakeRational: Optional[ Sequence[ bool ] ] = None
                 ) -> tuple[ tor.Tensor, NDArray[ np.str_ ], list[ int ] ]:
  '''Apply nonlinear functions elementwise to regressors with optional rationalisation.

  Transforms each regressor column through a list of ``NonLinearity`` objects.
  If ``MakeRational`` is True for a given function, its reciprocal form
  (multiplied by ``-y``) is also appended as a rational term.

  Parameters
  ----------
  y : torch.Tensor or None
      (p,) centred system output. Must be provided if ``MakeRational`` is not None.
  Data : torch.Tensor
      (p, n) tensor of regressors.
  RegNames : Sequence[str] or NDArray[np.str_]
      (n,) names of the regressor columns.
  Functions : Sequence[NonLinearity]
      List of nonlinearities to apply. The first element must be the identity ``"id"``.
  MakeRational : Sequence[bool], optional
      Per-function flag; if True, a rational term ``~ / f(reg)`` is appended.
      Must be None or match the length of ``Functions``.

  Returns
  -------
  Data : torch.Tensor
      (p, m) tensor with original, transformed, and rational columns concatenated.
  RegNames : NDArray[np.str_]
      (m,) updated column names reflecting the applied transformations.
  M : list[int]
      Indices mapping each output column to the index of the applied function.
  '''
  # ----------------------------------------------------------------------------- Bullshit prevention ------------------------------------------------------------------------------
  # Data tests
  if ( not isinstance( Data, tor.Tensor ) ): raise AssertionError( "The Input data must be a torch.Tensor" )
  if ( Data.ndim != 2 ):                     raise AssertionError( "The Input data 'Data' must be a 2D torch.Tensor. Reshape if single vector" )

  # Functions
  if ( not isinstance( Functions, list ) ):  raise AssertionError( "The 'Functions'argument name must be a list of NARMAX.NonLinearity objects" )
  if ( len( Functions ) == 0 ):              raise AssertionError( "The 'Functions'argument must be a non-empty list of NARMAX.NonLinearity objects" )
  if ( Functions[ 0 ].get_Name() != "id" ):  raise AssertionError( "The 0th function in the 'Functions' list must be 'id' per my convention that I decided. Thanks" )

  # MakeRational tests: flatten empty sequences to None to simplify procedure
  if ( ( MakeRational is not None ) and len( MakeRational ) == 0 ): MakeRational = None # use AND short circuiting to prevent type error for length

  if ( MakeRational is not None ): # check first since None has no length
    if ( len( Functions ) != len( MakeRational ) ): raise AssertionError( "The length of MakeRational doesn't match that of Functions" )
  else: # MakeRational is None, thus no rational fitting
    if ( len( Functions ) == 1 ): # Functions only contains "id" → identity
      print( "WARNING: No transformations (Functions) or MakeRational instructions were passed, which is sus as CTor.Lagger, thus, does nothing" )

  # RegNames tests
  if ( not isinstance( RegNames, np.ndarray ) ): RegNames = np.array( RegNames )
  if ( RegNames.ndim != 1 ):                     raise AssertionError( "The RegNames argument must be a 1D array of strings" )
  if ( len( RegNames ) != Data.shape[ 1 ] ):     raise AssertionError( "The RegNames argument must have the same length as Data's columns" )

  # y Tests
  if ( y is not None ):
    if ( not isinstance( y, tor.Tensor ) ):         raise AssertionError( "The 'y' argument must be a torch.Tensor" )
    if ( y.ndim == 0 ): raise AssertionError( "'y' must not be a scalar" )
    if ( y.ndim > 2 ):  raise AssertionError( "'y' must be 1D or a 2D tensor with a single row/column" )
    if ( y.ndim == 2 ):
      if ( ( y.shape[ 0 ] != 1 ) and ( y.shape[ 1 ] != 1 ) ): raise AssertionError( "'y' must be a column vector (p,1) or row vector (1,p), got shape {}".format( y.shape ) )
    y = y.view( -1 ) # flatten as security, since it's some transpose of 1D
    if ( len( y ) != Data.shape[ 0 ] ):                 raise AssertionError( "y's length does not match the Regressors' length" )

  else: # y is None
    if ( MakeRational is not None ):                    raise AssertionError( "y must be passed if MakeRational is not None" )

  # -------------------------------------------------------------------- A) Pre-Processing & B) Transformations --------------------------------------------------------------------

  # A) Set to list to easily append then concatenate data
  nRegs: int = Data.shape[ 1 ]
  DataList: list[ tor.Tensor ] = [ Data ]
  OutNames: list[ Sequence[ str ] ] = [ RegNames ]
  M: list[ int ] = [ 0 ] * nRegs # Morphing meta-data containing the index of the applied non-linearity (id for all un-processed terms)

  # B) Compute the transformations and append to the list to finally horizontally concatenate into a single matrix  
  for func in range( 1, len( Functions ) ): # start at 1 to ignore the identity function, skips the loop if Func only contains identity
    DataList.append( Functions[ func ].get_f()( DataList[ 0 ] ) ) # apply the function on the entire passed RegMat
    OutNames.append( [ Functions[ func ].get_Name() + "(" + col + ")" for col in OutNames[ 0 ] ] ) # Create the list of new names and apply directly
    M += [ func ] * nRegs # tag all regressors to have been processed with that function

  # ----------------------------------------------------------------------------- B) Rational Functions ----------------------------------------------------------------------------

  if ( ( MakeRational is not None ) and ( y is not None ) ): # y is guaranteed not None at this stage, but check to silence Pylance warning
    for func in range( len( Functions ) ):
      if ( MakeRational[ func ] ): # if the function is to be made rational (contains bool)
        DataList.append( - y.view( -1, 1 ) * DataList[ func ] ) # 1/ done via multiplication with -y
        M += [ 0 ] * nRegs # tag all rational terms as unmorphable at the moment

        RatNames: list[ str ] = [ None ] * nRegs # Init empty list of right length
        if ( Functions[ func ].get_Name() == "id" ): # equivalent to func == 0
          for col in range( len( RatNames ) ): RatNames[ col ] = "~/(" + OutNames[ 0 ][ col ] + ")" # ~/(Reg) for identity
        else:
          for col in range( len( RatNames ) ): RatNames[ col ] = "~/" + Functions[ func ].get_Name() + "(" + OutNames[ 0 ][ col ] + ")" # ~/func(Reg) for functions

        OutNames.append( RatNames ) # apply new column names

  return ( tor.hstack( DataList ), np.concatenate( OutNames ), M ) # Concatenate the list of segments into a single matrix


###################################################################################### Booler ######################################################################################
def Booler(
    Data: Union[ tor.Tensor, Sequence[ Sequence[ bool ] ] ],
    RegNames: Sequence[ str ],
    Operations: list[ Callable[ [ tor.Tensor, tor.Tensor ], tor.Tensor ] ] = [ tor.logical_and, tor.logical_xor, tor.logical_or ],
    OperationNames: Sequence[ str ] = [ "&&", "^", "||" ],
    AllowNegation: bool = True,
    ) -> tuple[ tor.Tensor, NDArray[ np.str_ ] ]:
  '''Creates boolean logic combinations of binary regressors.

  Generates original, negated, and pairwise binary combinations of the input
  regressors using the specified logical operations. Constant columns and
  duplicates are removed from the output.

  Parameters
  ----------
  Data : torch.Tensor or Sequence[Sequence[bool]]
      Binary input data, either as a 2D bool tensor (p, n) or a sequence
      of n sequences of length p.
  RegNames : Sequence[str]
      Names of each input binary regressor.
  Operations : list[Callable]
      List of binary logical operations to apply pairwise. Each callable
      must accept two bool tensors and return a bool tensor of the same shape. # R[1/2]
  OperationNames : Sequence[str]
      Display names for the corresponding operations (e.g. `"&&"`, `"^"`).
  AllowNegation : bool
      If True, negated versions of each regressor are also included.

  Returns
  -------
  OutMat : torch.Tensor
      (p, n_out) bool tensor of unique, non-constant binary combinations.
  OutNames : NDArray[np.str_]
      Array of column names describing each output column.
  '''

  # ---------- validation ----------
  def _validate_bool_binary_op( op: Callable[ [ tor.Tensor, tor.Tensor ], tor.Tensor ], name: str ) -> None:
    '''Validate that a binary boolean operation returns a bool tensor of the expected shape.'''
    a = tor.tensor( [ False, True, False, True, False, True, False, True ] )
    b = tor.tensor( [ False, False, True, True, False, False, True, True ] )
    try:
      out = op( a, b )
    except Exception as e:
      raise ValueError( f"Operation '{ name }' failed on test inputs: { e }" )

    if ( not isinstance( out, tor.Tensor ) ): raise ValueError( f"Operation '{ name }' did not return a tensor" )
    if ( out.shape != a.shape ):  raise ValueError( f"Operation '{ name }' changed input shape ({ a.shape } -> { out.shape })" )
    if ( out.dtype != tor.bool ): raise ValueError( f"Operation '{ name }' returned non-boolean dtype ({ out.dtype })" )

  if ( not isinstance( AllowNegation, bool ) ):             raise ValueError( "AllowNegation must be a bool" )
  if ( len( Operations ) != len( OperationNames ) ):          raise ValueError( "Operations and OperationNames must have the same length" )
  if ( not isinstance( Data, ( list, tuple, tor.Tensor ) ) ): raise ValueError( "Data must be a list, tuple, or torch.Tensor" )

  for op, name in zip( Operations, OperationNames ): _validate_bool_binary_op( op, name )

  if ( ( not Operations ) and ( not AllowNegation ) ): warnings.warn( "No operations and AllowNegation = False; no transformation applied." )

  # ---------- A) Convert to column‑stacked tensor ----------
  if ( isinstance( Data, ( list, tuple ) ) ): DataList = [ tor.as_tensor( col, dtype = tor.bool ) for col in Data ]
  else:                               DataList = [ Data[ :, i ] for i in range( Data.shape[ 1 ] ) ]

  if ( len( DataList ) != len( RegNames ) ): raise ValueError( "Data and RegNames must have the same number of regressors" )

  Orig = tor.column_stack( DataList ) # (p, n)
  Neg = tor.logical_not( Orig ) if AllowNegation else None

  # ---------- B) Determine maximum number of output columns ----------
  nBase = len( DataList ) * ( 2 if AllowNegation else 1 )
  nPairs = len( DataList ) * ( len( DataList ) - 1 ) // 2
  CombosPerPair = 4 if AllowNegation else 1
  maxCols = nBase + len( Operations ) * nPairs * CombosPerPair

  # Pre‑allocate the full output matrix and a matching name list
  OutMat = tor.empty( ( Orig.shape[ 0 ], maxCols ), dtype = tor.bool, device = Orig.device )
  OutNames = [ "" ] * maxCols
  CurrentCol = 0 # next free column index

  # ---------- C) Base columns (originals and optionally negations) ----------
  OutMat[ :, : len( DataList ) ] = Orig
  for i in range( len( DataList ) ): OutNames[ i ] = RegNames[ i ]
  CurrentCol = len( DataList )

  if ( AllowNegation ):
    OutMat[ :, CurrentCol : CurrentCol + len( DataList ) ] = Neg
    for i in range( len( DataList ) ): OutNames[ CurrentCol + i ] = "!" + RegNames[ i ]
    CurrentCol += len( DataList )

  # ---------- D) Pairwise combinations ----------
  for op_idx, op in enumerate( Operations ):
    for i in range( len( DataList ) - 1 ):

      L_orig = Orig[ :, i : i + 1 ] # original left: (p, 1)
      R_orig = Orig[ :, i + 1 : ] # original right block (p, n-i-1)

      # 1) L_orig  op  R_orig
      op( L_orig, R_orig, out = OutMat[ :, CurrentCol : CurrentCol + R_orig.shape[ 1 ] ] )
      for j in range( R_orig.shape[ 1 ] ): OutNames[ CurrentCol + j ] = f"{ RegNames[ i ] } { OperationNames[ op_idx ] } { RegNames[ i + 1 + j ] }"
      CurrentCol += R_orig.shape[ 1 ]

      if ( AllowNegation ):
        L_neg = Neg[ :, i : i + 1 ]
        R_neg = Neg[ :, i + 1 : ]

        # 2) L_neg  op  R_orig
        op( L_neg, R_orig, out = OutMat[ :, CurrentCol : CurrentCol + R_orig.shape[ 1 ] ] )
        for j in range( R_orig.shape[ 1 ] ): OutNames[ CurrentCol + j ] = f"!{ RegNames[ i ] } { OperationNames[ op_idx ] } { RegNames[ i + 1 + j ] }"
        CurrentCol += R_orig.shape[ 1 ]

        # 3) L_orig  op  R_neg
        op( L_orig, R_neg, out = OutMat[ :, CurrentCol : CurrentCol + R_orig.shape[ 1 ] ] )
        for j in range( R_orig.shape[ 1 ] ): OutNames[ CurrentCol + j ] = f"{ RegNames[ i ] } { OperationNames[ op_idx ] } !{ RegNames[ i + 1 + j ] }"
        CurrentCol += R_orig.shape[ 1 ]

        # 4) L_neg  op  R_neg
        op( L_neg, R_neg, out = OutMat[ :, CurrentCol : CurrentCol + R_orig.shape[ 1 ] ] )
        for j in range( R_orig.shape[ 1 ] ): OutNames[ CurrentCol + j ] = f"!{ RegNames[ i ] } { OperationNames[ op_idx ] } !{ RegNames[ i + 1 + j ] }"
        CurrentCol += R_orig.shape[ 1 ]

  # At this point we may have unused trailing columns if some were constant. Trim to the actually written columns.
  OutMat = OutMat[ :, : CurrentCol ]
  OutNames = OutNames[ : CurrentCol ]

  # ---------- E) Remove constant columns ----------
  col_sums = OutMat.sum( dim = 0 )
  keep = ( col_sums != 0 ) & ( col_sums != Orig.shape[ 0 ] )
  OutMat = OutMat[ :, keep ] # advanced indexing → copy
  OutNames = np.array( OutNames, dtype = str )[ keep.cpu().numpy() ]

  # ---------- F) Deduplicate (keep first occurrence) ----------
  if ( OutMat.shape[ 1 ] > 1 ):
    dedup_keep = tor.ones( OutMat.shape[ 1 ], dtype = tor.bool, device = Orig.device )
    for j in range( 1, OutMat.shape[ 1 ] ):
      matches = ( OutMat[ :, : j ] == OutMat[ :, j : j + 1 ] ).all( dim = 0 )
      if ( matches.any() ): dedup_keep[ j ] = False
    OutMat = OutMat[ :, dedup_keep ] # another copy of kept columns
    OutNames = OutNames[ dedup_keep.cpu().numpy() ]

  return OutMat, OutNames # R[2/2]

########################################################################## Ultra Orthogonal fitting Stuff ##########################################################################
def _superscript( n: int ) -> str:
  '''Convert an integer to its Unicode superscript representation.'''
  return str( n ).translate( str.maketrans( "0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹" ) )

# ************************************************************************* Differentiated Mollifier CTor *************************************************************************
def DiffedGaussianMollifier( FilterOrder: int = 31, nDerivatives: int = 2, std: float = 0.12, Plot: bool = False
                               ) -> list[ tor.Tensor ]:
  '''Generate FIR filter coefficients for a differentiated Gaussian mollifier.

  Constructs a set of FIR kernels corresponding to the Gaussian mollifier
  and its successive derivatives, used for smoothed derivative estimation.

  Parameters
  ----------
  FilterOrder : int
      Length of the FIR filter (number of taps). Must be >= 4.
  nDerivatives : int
      Number of derivative orders to compute (0 for just the Gaussian).
  std : float
      Standard deviation of the Gaussian kernel (relative to unit domain).
  Plot : bool
      If True, plots the generated filter coefficients.

  Returns
  -------
  MollifierFIRCoefficients : list[tor.Tensor]
      List of (FilterOrder,)-shaped tensors, where the i-th element
      corresponds to the i-th derivative of the Gaussian mollifier.
  '''
  if ( ( not isinstance( FilterOrder, int ) ) or ( FilterOrder < 1 ) ): raise ValueError( "FilterOrder must be a positive int" )
  if ( FilterOrder < 4 ): raise ValueError( "FilterOrder must be >= 4" )
  if ( ( not isinstance( nDerivatives, int ) ) or ( nDerivatives < 0 ) ): raise ValueError( "nDerivatives must be a non-negative int" )
  if ( not isinstance( std, ( int, float, np.floating, np.integer ) ) ): raise TypeError( "std must be numeric" )
  if ( std <= 0 ): raise ValueError( "std must be > 0" )
  std = float( std )

  X: np.ndarray = np.linspace( 0, 1, FilterOrder )
  Mu: float = 0.5
  S2: float = std * std

  G: np.ndarray = np.exp( -( X - Mu )**2 / ( 2 * S2 ) ) / ( std * np.sqrt( 2.0 * np.pi ) )

  MollifierFIRCoefficients: list[ np.ndarray ] = [ G ]
  Z: np.ndarray = -( X - Mu ) / S2
  if ( nDerivatives >= 1 ):
    MollifierFIRCoefficients.append( Z * MollifierFIRCoefficients[ 0 ] )
    for n in range( 2, nDerivatives + 1 ):
      MollifierFIRCoefficients.append( Z * MollifierFIRCoefficients[ -1 ] - ( ( n - 1.0 ) / S2 ) * MollifierFIRCoefficients[ -2 ] )

  SumG: float = float( np.sum( G ) )
  for i, D in enumerate( MollifierFIRCoefficients ):
    MollifierFIRCoefficients[ i ] = D / SumG
    if ( i >= 1 ):
      MollifierFIRCoefficients[ i ][ 0 ] = 0.0
      MollifierFIRCoefficients[ i ][ -1 ] = 0.0

  if ( Plot ):
    with plt.style.context( 'dark_background' ):
      Fig, Ax = plt.subplots()
      for Coeff in MollifierFIRCoefficients: Ax.plot( Coeff )
      Ax.legend( [ "∂⁰ Gaussian" ] + [ f"∂^{ i } Gaussian" for i in range( 1, nDerivatives + 1 ) ] )

  return [ tor.as_tensor( C ) for C in MollifierFIRCoefficients ]


# ************************************************************************************************ Derivative augment (vectorized) ********************************************************************************
def SmoothDeriver( Data: tor.Tensor, RegNames: Union[ Sequence[ str ], NDArray[ np.str_ ] ], nDerivatives: int = 2,
                   RegCoeffs: Optional[ Sequence[ float ] ] = None, FilterOrder: int = 31,
                   std: float = 0.12, Plot: bool = False, NormDerivatives: bool = True,
                   dt: Optional[ float ] = None
                 ) -> tuple[ tor.Tensor, NDArray[ np.str_ ] ]:
  '''Augment the regressor matrix with smoothed derivatives via differentiated mollifiers.

  Each column of the input data is convolved with the successive derivatives
  of a Gaussian mollifier to obtain smoothed derivative estimates, which are
  appended as new columns.

  Parameters
  ----------
  Data : torch.Tensor
      (p, n) tensor of regressor measurements.
  RegNames : Sequence[str] or NDArray[np.str_]
      Names of each input regressor column.
  nDerivatives : int
      Number of derivative orders to augment (must be >= 0).
  RegCoeffs : Sequence[float], optional
      Per-derivative scaling coefficients applied to each derivative order.
      If None, all are set to 1.0.
  FilterOrder : int
      Length of the mollifier FIR filter (must be >= 4).
  std : float
      Standard deviation of the Gaussian kernel.
  Plot : bool
      If True, plots the generated filter coefficients.
  NormDerivatives : bool
      If True, each derivative column is peak-normalised to unit amplitude.
  dt : float, optional
      Sampling interval in seconds. If given, derivative columns are scaled
      by (1 / (dt * (FilterOrder-1)))**order to obtain physical time derivatives.
      If None, raw normalised derivatives are returned.

  Returns
  -------
  OutMat : torch.Tensor
      (p, n * (1 + nDerivatives)) tensor with original data and derivative columns.
  OutNames : NDArray[np.str_]
      Array of column names, e.g. ``"∂² x"`` for the second derivative of ``x``.
  '''
  # --- unchanged input checks ---
  if ( not isinstance( Data, tor.Tensor ) ): raise TypeError( "Data must be a torch.Tensor" )
  if ( Data.ndim != 2 ):                   raise ValueError( "Data must be a 2D tensor (p, n)" )
  if ( not Data.dtype.is_floating_point ): raise TypeError( "Data dtype must be floating-point" )
  if ( ( not isinstance( nDerivatives, int ) ) or ( nDerivatives < 0 ) ): raise ValueError( "nDerivatives must be a non-negative int" )
  if ( ( not isinstance( FilterOrder, int ) ) or ( FilterOrder < 4 ) ): raise ValueError( "FilterOrder must be an int >= 4" )
  if ( not isinstance( std, ( int, float, np.floating, np.integer ) ) ): raise TypeError( "std must be numeric" )
  if ( std <= 0 ): raise ValueError( "std must be > 0" )
  if ( not isinstance( Plot, bool ) ): raise TypeError( "Plot must be a bool" )
  if ( not isinstance( NormDerivatives, bool ) ): raise TypeError( "NormDerivatives must be a bool" )
  if ( Data.shape[ 0 ] == 0 ): raise ValueError( "Data must have at least one sample (p >= 1)" )
  if ( not tor.isfinite( Data ).all() ): raise ValueError( "Data contains inf or NaN values" )
  if ( not isinstance( RegNames, np.ndarray ) ): RegNames = np.array( list( RegNames ) )
  if ( RegNames.ndim != 1 ): raise ValueError( "RegNames must be 1D" )
  if ( len( RegNames ) != Data.shape[ 1 ] ): raise ValueError( "Number of RegNames must match Data columns" )
  if ( dt is not None ):
    if ( not isinstance( dt, ( int, float, np.floating, np.integer ) ) ): raise TypeError( "dt must be numeric" )
    if ( dt <= 0 ): raise ValueError( "dt must be > 0" )
    dt = float( dt )

  if ( RegCoeffs is None ): RegCoeffs = [ 1.0 ] * nDerivatives
  else:                     RegCoeffs = list( RegCoeffs )
  if ( len( RegCoeffs ) != nDerivatives ): raise ValueError( f"RegCoeffs length ({ len( RegCoeffs ) }) must equal nDerivatives ({ nDerivatives })" )

  if ( Data.shape[ 1 ] == 0 ): return Data, np.array( [], dtype = np.str_ ) # R[1/2]

  # --- get FIR kernels ---
  FIRs: list[ tor.Tensor ] = [ fir.to( device = Data.device, dtype = Data.dtype ) for fir in  DiffedGaussianMollifier( FilterOrder, nDerivatives, std, Plot ) ]

  NOut: int = Data.shape[ 1 ] * ( 1 + nDerivatives )
  OutMat: tor.Tensor = tor.empty( ( Data.shape[ 0 ], NOut ), dtype = Data.dtype, device = Data.device )
  OutNames: NDArray[ np.str_ ] = np.empty( NOut, dtype = f'U{ max( len( str( n ) ) for n in RegNames ) + 2 + len( str( nDerivatives ) ) }' )

  # --- store original data ---
  OutMat[ :, : Data.shape[ 1 ] ] = Data
  for j in range( Data.shape[ 1 ] ): OutNames[ j ] = str( RegNames[ j ] )

  Half: int = FilterOrder // 2
  Window: tor.Tensor = tor.exp( tor.linspace( -4, 0, Half, device = Data.device, dtype = Data.dtype ) )

  Col: int = Data.shape[ 1 ]
  # Pre‑compute the base time‑scale denominator
  T_f = ( FilterOrder - 1 ) * dt if dt is not None else None

  for derivative in range( 1, nDerivatives + 1 ):
    Filt: tor.Tensor = FIRs[ derivative ].flip( 0 ).view( 1, 1, -1 ).expand( Data.shape[ 1 ], 1, -1 )
    Deriv: tor.Tensor = tor.nn.functional.conv1d( Data.T.unsqueeze( 0 ), Filt, padding = "same", groups = Data.shape[ 1 ] ).squeeze( 0 ).T

    # apply edge softening window
    if ( 0 < Half <= Deriv.shape[ 0 ] ):
      Deriv[ : Half ] *= Window.view( -1, 1 )
      Deriv[ -Half : ] *= Window.flip( 0 ).view( -1, 1 )

    # --- physical time scaling ---
    if ( dt is not None ): Deriv *= ( 1.0 / T_f )**derivative

    # --- optional peak normalisation (after scaling) ---
    if ( NormDerivatives ):
      for j in range( Deriv.shape[ 1 ] ):
        peak = Deriv[ :, j ].abs().max()
        if ( peak > 1e-15 ): Deriv[ :, j ] /= peak

    # --- apply per-derivative scaling coefficients ---
    Deriv *= RegCoeffs[ derivative - 1 ]

    OutMat[ :, Col : Col + Data.shape[ 1 ] ] = Deriv

    for j in range( Data.shape[ 1 ] ): OutNames[ Col + j ] = f"∂{ _superscript( derivative ) } { RegNames[ j ] }"
    Col += Data.shape[ 1 ]

  return OutMat, OutNames # R[2/2]
