import pytest
import warnings
import torch
import numpy as np
from typing import List, Union, Sequence

from NARMAX.CTors import Booler


# ----------------------------------------------------------------------
# Helpers for generating test data
# ----------------------------------------------------------------------
def make_data( p: int, n: int, as_tensor: bool = False ) -> Union[ torch.Tensor, List[ List[ bool ] ] ]:
  '''Return data with p rows and n columns. One column is all‑True, one all‑False.'''
  rng = np.random.RandomState( 42 )
  cols = []
  for i in range( n ):
    if ( i == 0 ): col = [ True ] * p # constant True
    elif ( i == 1 ): col = [ False ] * p # constant False
    else: col = rng.randint( 0, 2, size = p ).astype( bool ).tolist()
    cols.append( col )
  if ( as_tensor ): return torch.tensor( cols ).T # R[1/2] (p, n)
  return cols # R[2/2]


def reg_names( n: int ) -> List[ str ]: return [ f"x{ i }" for i in range( n ) ]


# ----------------------------------------------------------------------
# Operation validation tests
# ----------------------------------------------------------------------
def test_validate_op_good() -> None:
  '''Valid operation parameters do not raise.'''
  res, names = Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
                        Operations = [ torch.logical_and ], OperationNames = [ "&" ], AllowNegation = False )
  assert isinstance( res, torch.Tensor )
  assert res.dtype == torch.bool
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U'


def test_validate_op_bad_type() -> None:
  '''Bad operation return type raises ValueError.'''
  with pytest.raises( ValueError, match = "did not return a tensor" ):
    Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
               Operations = [ lambda a, b : True ], OperationNames = [ "bad" ], AllowNegation = False )


def test_validate_op_wrong_shape() -> None:
  '''Operation changing input shape raises ValueError.'''
  with pytest.raises( ValueError, match = "changed input shape" ):
    Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
               Operations = [ lambda a, b : a[ 0 : 1 ] ], OperationNames = [ "bad" ], AllowNegation = False )


def test_validate_op_non_bool_dtype() -> None:
  '''Operation returning non-bool dtype raises ValueError.'''
  with pytest.raises( ValueError, match = "non-boolean dtype" ):
    Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
               Operations = [ lambda a, b : a.to( torch.int8 ) + b ], OperationNames = [ "add" ], AllowNegation = False )


# ----------------------------------------------------------------------
# Input validation and edge cases
# ----------------------------------------------------------------------
@pytest.mark.parametrize( "bad_value", [ "yes", 1, 0, 3.14, None, [ True ] ] )
def test_allow_negation_not_bool( bad_value ) -> None:
  '''Non-bool AllowNegation raises ValueError.'''
  with pytest.raises( ValueError, match = "AllowNegation must be a bool" ):
    Booler( torch.tensor( [ [ True ] ] ), [ "a" ], AllowNegation = bad_value )


def test_operations_names_mismatch() -> None:
  '''Mismatched Operations/OperationNames length raises ValueError.'''
  with pytest.raises( ValueError, match = "Operations and OperationNames must have the same length" ):
    Booler( torch.tensor( [ [ True ] ] ), [ "a" ], Operations = [ torch.logical_and ], OperationNames = [ "&&", "||" ] )


@pytest.mark.parametrize( "bad_data", [ "not a tensor", 42, 3.14, None, { "a": True }, { True, False } ] )
def test_invalid_data_type( bad_data ) -> None:
  '''Invalid data type raises ValueError.'''
  with pytest.raises( ValueError, match = "Data must be a list, tuple, or torch.Tensor" ):
    Booler( bad_data, [ "a" ] )


def test_data_regnames_column_count_mismatch() -> None:
  '''Column/RegNames count mismatch raises ValueError.'''
  data = torch.tensor( [ [ True, False ] ] )
  with pytest.raises( ValueError, match = "Data and RegNames must have the same number" ):
    Booler( data, [ "only_one" ] )


def test_no_ops_no_negation_warns( capsys ) -> None:
  '''No operations and no negation emits warning.'''
  data = torch.tensor( [ [ True, False ] ] )
  with pytest.warns( UserWarning, match = "No operations and AllowNegation = False" ):
    Booler( data, [ "a", "b" ], Operations = [], OperationNames = [], AllowNegation = False )
  # Ensure the warning was captured
  # capsys not needed for pytest.warns, but we can also check that output is original (minus constants)
  res, names = Booler( data, [ "a", "b" ], Operations = [], OperationNames = [], AllowNegation = False )
  # Original data has a constant True column (col0) and constant False column (col1) – both removed
  assert res.shape == ( 1, 0 )
  assert len( names ) == 0


def test_no_ops_negation_true() -> None:
  '''Negation without operations works correctly.'''
  data = torch.tensor( [ [ True, False ] ] )
  # No warning expected
  res, names = Booler( data, [ "a", "b" ], Operations = [], OperationNames = [], AllowNegation = True )
  # Original (True,False) + negated (False,True) -> after constant removal:
  # True col -> all True (removed), False -> all False (removed)
  # negated: original True becomes False (removed), original False becomes True (removed)
  # All columns are constant → empty output
  assert res.shape == ( 1, 0 )
  assert len( names ) == 0
  assert res.dtype == torch.bool


# ----------------------------------------------------------------------
# Basic functionality
# ----------------------------------------------------------------------
@pytest.mark.parametrize( "as_tensor", [ True, False ] )
def test_single_column_no_negation( as_tensor: bool ) -> None:
  '''Single column without negation.'''
  data_in = make_data( 5, 1, as_tensor = as_tensor )
  # Single column, constant True -> will be removed after filtering
  res, names = Booler( data_in, [ "x0" ], Operations = [ torch.logical_and ], OperationNames = [ "&&" ], AllowNegation = False )
  # With 1 column, no pairs -> no extra ops. Original column is constant True -> removed
  assert res.numel() == 0 or res.shape[ 1 ] == 0
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U'
  if ( res.numel() > 0 ):
    assert res.dtype == torch.bool


@pytest.mark.parametrize( "as_tensor", [ True, False ] )
def test_single_column_with_negation( as_tensor: bool ) -> None:
  '''Single column with negation.'''
  data_in = make_data( 5, 1, as_tensor = as_tensor ) # constant True
  res, names = Booler( data_in, [ "x0" ], Operations = [ torch.logical_and ], OperationNames = [ "&&" ], AllowNegation = True )
  # Original True (constant removed), negated False (constant removed) => empty
  assert res.shape[ 1 ] == 0
  assert res.dtype == torch.bool
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U'


def test_two_columns_no_negation_basic_combos() -> None:
  '''Two columns without negation produce correct combos.'''
  data = torch.tensor( [ [ True, True ],
                         [ True, False ],
                         [ False, True ],
                         [ False, False ] ] )
  res, names = Booler( data, [ "A", "B" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = False )
  # Expected: A, B, A & B (originals included, no duplicates), shape (4,3)
  assert res.shape == ( 4, 3 )
  assert names.tolist() == [ "A", "B", "A & B" ]
  assert torch.equal( res[ :, 0 ], data[ :, 0 ] ) # A
  assert torch.equal( res[ :, 1 ], data[ :, 1 ] ) # B
  expected_and = torch.logical_and( data[ :, 0 ], data[ :, 1 ] )
  assert torch.equal( res[ :, 2 ], expected_and ) # A & B


def test_two_columns_with_negation_all_combos() -> None:
  '''Two columns with negation produce all 8 combos.'''
  data = torch.tensor( [ [ True, True ],
                         [ True, False ],
                         [ False, True ],
                         [ False, False ] ] )
  res, names = Booler( data, [ "A", "B" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = True )
  # We should have: A, B, !A, !B + four combos: A&B, !A&B, A&!B, !A&!B
  # All eight are non-constant and unique (check manually)
  expected_cols = {
        "A":           torch.tensor( [ True, True, False, False ] ),
        "B":           torch.tensor( [ True, False, True, False ] ),
        "!A":          torch.tensor( [ False, False, True, True ] ),
        "!B":          torch.tensor( [ False, True, False, True ] ),
        "A & B":       torch.tensor( [ True, False, False, False ] ),
        "!A & B":      torch.tensor( [ False, False, True, False ] ),
        "A & !B":      torch.tensor( [ False, True, False, False ] ),
        "!A & !B":     torch.tensor( [ False, False, False, True ] ),
    }
  assert res.shape == ( 4, 8 )
  for i, name in enumerate( names ): assert torch.equal( res[ :, i ], expected_cols[ name ] )


def test_multiple_operations() -> None:
  '''Multiple operations applied correctly.'''
  # 4 rows so that A&B and A|B are both non-constant
  data = torch.tensor( [ [ True, False ],
                         [ False, True ],
                         [ True, True ],
                         [ False, False ] ] )
  ops = [ torch.logical_and, torch.logical_or ]
  op_names = [ "&", "|" ]
  res, names = Booler( data, [ "A", "B" ], Operations = ops, OperationNames = op_names,
                        AllowNegation = False )
  # Columns: A, B, A&B, A|B (originals included, all unique)
  assert res.shape == ( 4, 4 )
  assert names.tolist() == [ "A", "B", "A & B", "A | B" ]
  assert torch.equal( res[ :, 0 ], torch.tensor( [ True, False, True, False ] ) ) # A
  assert torch.equal( res[ :, 1 ], torch.tensor( [ False, True, True, False ] ) ) # B
  assert torch.equal( res[ :, 2 ], torch.tensor( [ False, False, True, False ] ) ) # A&B
  assert torch.equal( res[ :, 3 ], torch.tensor( [ True, True, True, False ] ) ) # A|B


# ----------------------------------------------------------------------
# Constant column removal
# ----------------------------------------------------------------------
def test_constant_columns_removed() -> None:
  '''Constant columns are removed from output.'''
  data = torch.tensor( [ [ True, False, True ],
                         [ True, False, False ] ] )
  # Columns: 0 all True, 1 all False, 2 mixed
  res, names = Booler( data, [ "x0", "x1", "x2" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&&" ], AllowNegation = False )
  # Originals: x0 (constant True, removed), x1 (constant False, removed), x2 (mixed, kept)
  # Pairs: x0&&x1 (constant, removed), x0&&x2 = x2 (duplicate of leftward x2, removed), x1&&x2 (constant, removed)
  # So only x2 survives
  assert names.tolist() == [ "x2" ]
  assert torch.equal( res[ :, 0 ], torch.tensor( [ True, False ] ) )


def test_all_constant_output() -> None:
  '''All-constant input results in empty output.'''
  # Data has only two columns, both constant True
  data = torch.tensor( [ [ True, True ],
                         [ True, True ] ] )
  res, names = Booler( data, [ "A", "B" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&&" ], AllowNegation = True )
  # All possible combos will be constant (True or False) -> everything removed
  assert res.shape[ 1 ] == 0
  assert len( names ) == 0
  assert res.dtype == torch.bool


# ----------------------------------------------------------------------
# Names output format
# ----------------------------------------------------------------------
def test_negation_names() -> None:
  '''Negation produces correctly named columns and values.'''
  # Use 4 rows so that columns are not all-constant
  data = torch.tensor( [ [ True, True ],
                         [ True, False ],
                         [ False, True ],
                         [ False, False ] ] )
  res, names = Booler( data, [ "x", "y" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = True )
  assert set( names ) == { "x", "y", "!x", "!y", "x & y", "!x & y", "x & !y", "!x & !y" }
  expected_cols = {
    "x":       torch.tensor( [ True, True, False, False ] ),
    "y":       torch.tensor( [ True, False, True, False ] ),
    "!x":      torch.tensor( [ False, False, True, True ] ),
    "!y":      torch.tensor( [ False, True, False, True ] ),
    "x & y":   torch.tensor( [ True, False, False, False ] ),
    "!x & y":  torch.tensor( [ False, False, True, False ] ),
    "x & !y":  torch.tensor( [ False, True, False, False ] ),
    "!x & !y": torch.tensor( [ False, False, False, True ] ),
  }
  for i, name in enumerate( names ): assert torch.equal( res[ :, i ], expected_cols[ name ] )


def test_names_are_ndarray_str() -> None:
  '''Output names are a unicode numpy array.'''
  _, names = Booler( torch.tensor( [ [ True ] ] ), [ "a" ], AllowNegation = False )
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U' # unicode string


# ----------------------------------------------------------------------
# Output dtype and shape
# ----------------------------------------------------------------------
def test_output_dtype() -> None:
  '''Output tensor has bool dtype.'''
  res, _ = Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
                    Operations = [ torch.logical_and ], OperationNames = [ "&&" ], AllowNegation = False )
  assert res.dtype == torch.bool


def test_output_shape_rows_unchanged() -> None:
  '''Number of rows is preserved.'''
  p = 10
  data = torch.randint( 0, 2, ( p, 5 ) ).bool()
  res, _ = Booler( data, [ f"c{ i }" for i in range( 5 ) ], AllowNegation = True )
  assert res.shape[ 0 ] == p


# ----------------------------------------------------------------------
# Large random integration test
# ----------------------------------------------------------------------
def test_large_random_integration() -> None:
  '''Large random inputs produce correct outputs.'''
  p, n = 100, 6
  rng = np.random.RandomState( 123 )
  cols = [ rng.randint( 0, 2, size = p ).astype( bool ).tolist() for _ in range( n ) ]
  res, names = Booler( cols, reg_names( n ), Operations = [ torch.logical_and, torch.logical_or ],
                        OperationNames = [ "&", "|" ], AllowNegation = True )
  # Should not crash; no column constant (probability near 0)
  assert res.shape[ 0 ] == p
  assert len( names ) > 0
  assert res.dtype == torch.bool
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U'
  # Spot-check first column matches first regressor
  expected_first = torch.tensor( cols[ 0 ], dtype = torch.bool )
  assert torch.equal( res[ :, 0 ], expected_first )


# ----------------------------------------------------------------------
# Additional edge cases
# ----------------------------------------------------------------------
def test_data_with_duplicate_names_constant_true_false() -> None:
  '''Input with a single constant True column produces empty output.'''
  data = torch.tensor( [ [ True ], [ True ] ] )
  res, names = Booler( data, [ "x" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = False )
  assert res.shape == ( 2, 0 )
  assert len( names ) == 0
  assert res.dtype == torch.bool
  assert isinstance( names, np.ndarray )


def test_duplicate_columns_removed() -> None:
  '''Columns identical to earlier columns are deduplicated.'''
  data = torch.tensor( [ [ True, True ],
                         [ False, False ] ] )
  # A == B, and A & B == A
  res, names = Booler( data, [ "A", "B" ], Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = False )
  assert names.tolist() == [ "A" ]
  assert torch.equal( res[ :, 0 ], data[ :, 0 ] )
  assert res.dtype == torch.bool


def test_regnames_as_tuple() -> None:
  '''Passing RegNames as a tuple works correctly.'''
  data = torch.tensor( [ [ True, True ],
                         [ True, False ],
                         [ False, True ],
                         [ False, False ] ] )
  res, names = Booler( data, ( "a", "b" ), Operations = [ torch.logical_and ],
                        OperationNames = [ "&" ], AllowNegation = False )
  assert isinstance( names, np.ndarray )
  assert names.tolist() == [ "a", "b", "a & b" ]
  assert torch.equal( res[ :, 0 ], data[ :, 0 ] )


def test_operation_not_callable() -> None:
  '''Non-callable operation raises ValueError.'''
  with pytest.raises( ValueError, match = "failed on test inputs" ):
    Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
            Operations = [ "not_callable" ], OperationNames = [ "bad" ], AllowNegation = False )


def test_operation_wrong_arg_count() -> None:
  '''Operation with wrong argument count raises ValueError.'''
  with pytest.raises( ValueError, match = "failed on test inputs" ):
    Booler( torch.tensor( [ [ True, False ] ] ), [ "a", "b" ],
            Operations = [ lambda x: x ], OperationNames = [ "bad" ], AllowNegation = False )


# ----------------------------------------------------------------------
# Edge: p=0 (empty rows)
# ----------------------------------------------------------------------
def test_zero_rows() -> None:
  '''Zero rows produce empty output.'''
  data = torch.empty( 0, 3, dtype = torch.bool )
  res, names = Booler( data, [ "a", "b", "c" ], AllowNegation = True )
  assert res.shape[ 0 ] == 0
  # All columns are constant (all True/False), so they will be removed
  assert res.shape[ 1 ] == 0
  assert len( names ) == 0
  assert res.dtype == torch.bool
  assert isinstance( names, np.ndarray )
  assert names.dtype.kind == 'U'
