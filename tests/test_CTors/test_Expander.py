import torch
import numpy as np
import pytest
from typing import List

from NARMAX.CTors import Expander

# Helper to create a simple 2D tensor
def make_data( *cols: List ) -> torch.Tensor:
  '''cols: each is a 1D list -> makes a tensor with those as columns.'''
  return torch.tensor( list( zip( *cols ) ), dtype = torch.float32 )


def test_order1_returns_identity() -> None:
  '''Order 1 returns original tensor.'''
  x = make_data( [ 1, 2, 3 ], [ 4, 5, 6 ] )
  names = [ 'u', 'v' ]
  M, nms = Expander( x, names, 1 )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M is x # returns the same tensor object
  assert nms.tolist() == names
  assert M.shape == ( 3, 2 )
  assert M.dtype == x.dtype


def test_order1_empty() -> None:
  '''Order 1 with empty data works.'''
  x = torch.empty( 5, 0, dtype = torch.float32 )
  M, nms = Expander( x, [], 1 )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert M.dtype == x.dtype
  assert M.shape == ( 5, 0 )
  assert len( nms ) == 0


def test_order2_full_simple() -> None:
  '''Order 2 full expansion works correctly.'''
  # one row to keep it clear
  x = torch.tensor( [ [ 2.0, 3.0 ] ] ) # u=2, v=3
  M, nms = Expander( x, [ 'u', 'v' ], 2, IteractionOnly = False )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  # Expected: u, v, u^2, u*v, v^2
  expected_M = torch.tensor( [ [ 2.0, 3.0, 4.0, 6.0, 9.0 ] ] )
  expected_names = np.array( [ 'u', 'v', 'u^2', 'u * v', 'v^2' ] )
  assert torch.allclose( M, expected_M )
  assert np.array_equal( nms, expected_names )


def test_order2_interaction_only_simple() -> None:
  '''Order 2 interaction-only expansion works correctly.'''
  x = torch.tensor( [ [ 2.0, 3.0 ] ] )
  M, nms = Expander( x, [ 'u', 'v' ], 2, IteractionOnly = True )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  # Expected: u, v, u*v
  expected_M = torch.tensor( [ [ 2.0, 3.0, 6.0 ] ] )
  expected_names = np.array( [ 'u', 'v', 'u * v' ] )
  assert torch.allclose( M, expected_M )
  assert np.array_equal( nms, expected_names )


def test_order3_full_three_vars() -> None:
  '''Order 3 full expansion with three variables.'''
  x = torch.tensor( [ [ 1.0, 2.0, 3.0 ] ] ) # a, b, c
  M, nms = Expander( x, [ 'a', 'b', 'c' ], 3, IteractionOnly = False )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  # Number of terms: C(3+3,3)-1 = 20-1=19
  assert M.shape == ( 1, 19 )
  assert len( nms ) == 19
  # Check a few known values
  # order 1: a=1, b=2, c=3
  assert M[ 0, 0 ] == 1.0
  assert M[ 0, 1 ] == 2.0
  assert M[ 0, 2 ] == 3.0
  # order 2: a^2=1, a*b=2, a*c=3, b^2=4, b*c=6, c^2=9
  # exact positions after deg1: col 3..8
  assert M[ 0, 3 ] == 1.0 # a^2
  assert M[ 0, 4 ] == 2.0 # a*b
  assert M[ 0, 5 ] == 3.0 # a*c
  assert M[ 0, 6 ] == 4.0 # b^2
  assert M[ 0, 7 ] == 6.0 # b*c
  assert M[ 0, 8 ] == 9.0 # c^2
  # order 3: C(3+3-1,3)=10 terms
  # Manually check a^3=1, a^2*b=2, a^2*c=3, a*b^2=4, a*b*c=6, a*c^2=9,
  # b^3=8, b^2*c=12, b*c^2=18, c^3=27
  expected_order3 = [ 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0 ]
  for i, val in enumerate( expected_order3 ): assert M[ 0, 9 + i ] == val, f"Mismatch at order3 term { i }"

  # Check names at a few positions
  assert nms[ 3 ] == 'a^2'
  assert nms[ 8 ] == 'c^2'
  assert nms[ 9 ] == 'a^3'
  assert nms[ -1 ] == 'c^3'


def test_order3_interaction_only_three_vars() -> None:
  '''Order 3 interaction-only with three variables.'''
  x = torch.tensor( [ [ 1.0, 2.0, 3.0 ] ] )
  M, nms = Expander( x, [ 'a', 'b', 'c' ], 3, IteractionOnly = True )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  # Expect: a,b,c, a*b, a*c, b*c, a*b*c => total 7 terms
  assert M.shape == ( 1, 7 )
  expected_values = [ 1., 2., 3., 2., 3., 6., 6. ] # a,b,c,ab,ac,bc,abc
  assert torch.allclose( M, torch.tensor( [ expected_values ] ) )
  expected_names = np.array( [ 'a', 'b', 'c', 'a * b', 'a * c', 'b * c', 'a * b * c' ] )
  assert np.array_equal( nms, expected_names )


def test_multi_row_preserves_values() -> None:
  '''Multi-row data produces correct values per row.'''
  x = torch.tensor( [ [ 1.0, 2.0 ], [ 3.0, 4.0 ] ] )
  M, nms = Expander( x, [ 'u', 'v' ], 2, IteractionOnly = False )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  assert M.device == x.device
  assert M.shape == ( 2, 5 )
  # Row 0: u=1, v=2 -> 1, 2, 1, 2, 4
  assert torch.allclose( M[ 0 ], torch.tensor( [ 1.0, 2.0, 1.0, 2.0, 4.0 ] ) )
  # Row 1: u=3, v=4 -> 3, 4, 9, 12, 16
  assert torch.allclose( M[ 1 ], torch.tensor( [ 3.0, 4.0, 9.0, 12.0, 16.0 ] ) )


def test_order_exceeds_nvars_interaction() -> None:
  '''When order > n_cols, interaction‑only stops earlier.'''
  x = torch.tensor( [ [ 1.0, 2.0 ] ] ) # 2 vars
  M, nms = Expander( x, [ 'u', 'v' ], 5, IteractionOnly = True )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  # Can only have up to order 2 (pair). So terms: u, v, u*v => 3
  assert M.shape == ( 1, 3 )
  assert M[ 0, 0 ] == 1.0
  assert M[ 0, 1 ] == 2.0
  assert M[ 0, 2 ] == 2.0
  assert M.dtype == x.dtype
  assert nms.tolist() == [ 'u', 'v', 'u * v' ]


def test_order_exceeds_nvars_full() -> None:
  '''Full expansion works when order exceeds n_cols.'''
  x = torch.tensor( [ [ 2.0, 3.0 ] ] )
  M, nms = Expander( x, [ 'u', 'v' ], 4, IteractionOnly = False )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.dtype == x.dtype
  # Number of terms = C(2+4,4)-1 = C(6,4)-1 = 15-1=14? Actually C(6,2)-1=15-1=14? Wait C(2+4,4)=C(6,4)=15, -1=14.
  assert M.shape == ( 1, 14 )
  # Check a high‑order value: u^2 v^2 = (2^2)*(3^2)=36
  # Need to find the column index. We'll verify existence via name.
  assert 'u^2 * v^2' in nms
  idx = list( nms ).index( 'u^2 * v^2' )
  assert M[ 0, idx ] == 36.0


def test_device_awareness() -> None:
  '''GPU tensors stay on GPU.'''
  if ( torch.cuda.is_available() ):
    x = torch.tensor( [ [ 1.0, 2.0 ] ], device = 'cuda' )
    M, _ = Expander( x, [ 'u', 'v' ], 2 )
    assert M.device == x.device
    assert M.dtype == x.dtype


def test_dtype_preserved() -> None:
  '''Input dtype is preserved in output.'''
  x = torch.tensor( [ [ 1.0, 2.0 ] ], dtype = torch.float64 )
  M, _ = Expander( x, [ 'u', 'v' ], 2 )
  assert M.dtype == torch.float64
  assert M.device == x.device


def test_regnames_as_list_vs_ndarray() -> None:
  '''List and array RegNames produce same result.'''
  x = make_data( [ 1, 2 ], [ 3, 4 ] )
  M1, nms1 = Expander( x, [ 'x', 'y' ], 2 )
  M2, nms2 = Expander( x, np.array( [ 'x', 'y' ] ), 2 )
  assert isinstance( nms1, np.ndarray )
  assert isinstance( nms2, np.ndarray )
  assert nms1.dtype.kind == 'U'
  assert nms2.dtype.kind == 'U'
  assert torch.equal( M1, M2 )
  assert np.array_equal( nms1, nms2 )


def test_expansion_order_type() -> None:
  '''ExpansionOrder type validation.'''
  x = make_data( [ 1 ] )
  # numpy integer should be accepted
  M, _ = Expander( x, [ 'a' ], np.int32( 1 ) )
  assert M is x
  # float should raise
  with pytest.raises( ValueError, match = "int >= 1" ):
    Expander( x, [ 'a' ], 1.0 )
  # negative int
  with pytest.raises( ValueError, match = "int >= 1" ):
    Expander( x, [ 'a' ], -1 )
  # zero should raise (ExpansionOrder < 1)
  with pytest.raises( ValueError, match = "int >= 1" ):
    Expander( x, [ 'a' ], 0 )
  # bool True is subclass of int and equals 1, so should work as order 1
  M, _ = Expander( x, [ 'a' ], True )
  assert M is x


def test_data_dimension_errors() -> None:
  '''Non-2D input raises ValueError.'''
  x = torch.tensor( [ 1.0, 2.0 ] ) # 1D
  with pytest.raises( ValueError, match = "2-dimensional" ):
    Expander( x, [ 'a' ], 2 )
  # 3D input also raises
  x3d = torch.tensor( [ [ [ 1.0 ] ] ] )
  with pytest.raises( ValueError, match = "2-dimensional" ):
    Expander( x3d, [ 'a' ], 2 )


def test_names_length_mismatch() -> None:
  '''Name count mismatch raises ValueError.'''
  x = make_data( [ 1, 2 ], [ 3, 4 ] )
  with pytest.raises( ValueError, match = "Number of names" ):
    Expander( x, [ 'a' ], 2 )


def test_no_nan_remaining() -> None:
  '''No NaN values in expanded output.'''
  x = torch.rand( 5, 3 )
  M, nms = Expander( x, [ 'a', 'b', 'c' ], 3, IteractionOnly = False )
  assert isinstance( M, torch.Tensor )
  assert isinstance( nms, np.ndarray )
  assert nms.dtype.kind == 'U'
  assert M.shape[ 0 ] == x.shape[ 0 ]
  assert not torch.isnan( M ).any()
  assert M.dtype == x.dtype


def test_input_not_mutated() -> None:
  '''Original tensor is not mutated by Expander.'''
  x = torch.tensor( [ [ 2.0, 3.0 ] ] )
  x_copy = x.clone()
  Expander( x, [ 'u', 'v' ], 3, IteractionOnly = False )
  assert torch.equal( x, x_copy )


def test_degenerate_iteraction_no_possible_terms() -> None:
  '''Single variable interaction-only produces just the variable.'''
  x = torch.tensor( [ [ 5.0 ] ] )
  for order in [ 1, 2, 5 ]:
    M, nms = Expander( x, [ 'z' ], order, IteractionOnly = True )
    assert isinstance( M, torch.Tensor )
    assert isinstance( nms, np.ndarray )
    assert nms.dtype.kind == 'U'
    assert M.shape == ( 1, 1 )
    assert M[ 0, 0 ] == 5.0
    assert M.dtype == x.dtype
    assert nms[ 0 ] == 'z'
