import pytest
import numpy as np
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )
from NARMAX.Classes.MultiKeyHashTable import MultiKeyHashTable


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def EmptyTable() -> MultiKeyHashTable: return MultiKeyHashTable()


@pytest.fixture
def TableInt64() -> MultiKeyHashTable: return MultiKeyHashTable( IntType = np.int64 )


@pytest.fixture
def TableWithData() -> MultiKeyHashTable:
  table = MultiKeyHashTable( IntType = np.int64 )
  a1 = np.array( [ 3, 1, 5, 7 ], dtype = np.int64 )
  a2 = np.array( [ 2, 4, 6 ], dtype = np.int64 )
  a3 = np.array( [ 10, 20 ], dtype = np.int64 )
  idx1 = table.AddData( a1 )
  idx2 = table.AddData( a2 )
  idx3 = table.AddData( a3 )
  table.CreateKeys( MinLen = 1, IndexSet = a1, Value = idx1 )
  table.CreateKeys( MinLen = 2, IndexSet = a2, Value = idx2 )
  table.CreateKeys( MinLen = 1, IndexSet = a3, Value = idx3 )
  return table


# ----------------------------------------------------------------------
# Tests for __init__
# ----------------------------------------------------------------------
def test_init_empty( EmptyTable ) -> None:
  '''New instance initialises without error and has no data.'''
  assert EmptyTable.DataCount == 0


# ----------------------------------------------------------------------
# Tests for AddData
# ----------------------------------------------------------------------
def test_add_data_returns_correct_index( TableInt64 ) -> None:
  '''Adding data returns sequential indices starting from 0.'''
  item = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = TableInt64.AddData( item )
  assert isinstance( idx, int )
  assert idx == 0
  assert np.array_equal( TableInt64[ 0 ], item )
  assert TableInt64[ 0 ].dtype == np.int64

  item2 = np.array( [ 4, 5 ], dtype = np.int64 )
  idx2 = TableInt64.AddData( item2 )
  assert isinstance( idx2, int )
  assert idx2 == 1
  assert np.array_equal( TableInt64[ 1 ], item2 )
  assert TableInt64[ 1 ].dtype == np.int64


def test_add_data_stores_exact_array( TableInt64 ) -> None:
  '''Modifying the original does not affect storage (a copy is stored).'''
  item = np.array( [ 42, 7 ], dtype = np.int64 )
  idx = TableInt64.AddData( item )
  assert TableInt64[ idx ].dtype == np.int64
  item[ 0 ] = 99
  item[ 1 ] = -1
  assert np.array_equal( TableInt64[ idx ], np.array( [ 42, 7 ] ) )


def test_add_data_with_empty_array( TableInt64 ) -> None:
  '''An empty array can be added and retrieved.'''
  empty_arr = np.array( [], dtype = np.int64 )
  idx = TableInt64.AddData( empty_arr )
  assert TableInt64[ idx ].size == 0
  assert TableInt64[ idx ].dtype == np.int64
  assert np.array_equal( TableInt64[ idx ], empty_arr )


def test_add_data_twice_same_array( TableInt64 ) -> None:
  '''Adding the same array twice stores two separate copies.'''
  arr = np.array( [ 7, 8 ], dtype = np.int64 )
  idx1 = TableInt64.AddData( arr )
  idx2 = TableInt64.AddData( arr )
  assert idx2 == idx1 + 1
  assert np.array_equal( TableInt64[ idx1 ], arr )
  assert np.array_equal( TableInt64[ idx2 ], arr )


# ----------------------------------------------------------------------
# Tests for __getitem__
# ----------------------------------------------------------------------
def test_getitem_returns_correct_array( TableInt64 ) -> None:
  '''Indexing the table with a valid index returns the correct array.'''
  a = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  b = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  idx_a = TableInt64.AddData( a )
  idx_b = TableInt64.AddData( b )
  assert np.array_equal( TableInt64[ idx_a ], a )
  assert TableInt64[ idx_a ].dtype == np.int64
  assert np.array_equal( TableInt64[ idx_b ], b )
  assert TableInt64[ idx_b ].dtype == np.int64


def test_getitem_raises_index_error( EmptyTable ) -> None:
  '''Indexing with an out-of-range index raises IndexError.'''
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = EmptyTable[ 0 ]
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = EmptyTable[ -1 ]
  EmptyTable.AddData( np.array( [ 1 ], dtype = np.int64 ) )
  with pytest.raises( IndexError ):
    _ = EmptyTable[ 1 ]
  with pytest.raises( IndexError ):
    _ = EmptyTable[ -2 ]


def test_getitem_after_deletion_still_works( TableInt64 ) -> None:
  '''Deleting keys does not delete the underlying data.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = TableInt64.AddData( seq )
  TableInt64.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  TableInt64.DeleteAllOfSize( 3 )
  assert np.array_equal( TableInt64[ idx ], seq )
  assert TableInt64[ idx ].dtype == np.int64


def test_getitem_negative_index( TableInt64 ) -> None:
  '''Negative indices access elements from the end of the data storage.'''
  a = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  b = np.array( [ 40, 50 ], dtype = np.int64 )
  idx_a = TableInt64.AddData( a )
  idx_b = TableInt64.AddData( b )

  assert np.array_equal( TableInt64[ -1 ], b )
  assert TableInt64[ -1 ].dtype == np.int64
  assert np.array_equal( TableInt64[ -2 ], a )
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = TableInt64[ -3 ]


# ----------------------------------------------------------------------
# Tests for SameStart
# ----------------------------------------------------------------------
def test_samestart_exact_match( TableWithData ) -> None:
  '''A query matching a stored prefix returns the correct data index.'''
  query = np.array( [ 5, 1, 3 ], dtype = np.int64 )
  result = TableWithData.SameStart( query )
  assert isinstance( result, int )
  assert result == 0


def test_samestart_subset_match( TableWithData ) -> None:
  '''A query matching a subset prefix returns the correct data index.'''
  query = np.array( [ 4, 2 ], dtype = np.int64 )
  result = TableWithData.SameStart( query )
  assert isinstance( result, int )
  assert result == 1


def test_samestart_no_match( EmptyTable ) -> None:
  '''A query that does not match any prefix returns None.'''
  EmptyTable.AddData( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  query = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  result = EmptyTable.SameStart( query )
  assert result is None


def test_samestart_empty_query( EmptyTable ) -> None:
  '''An empty query returns None (no zero-length keys exist).'''
  EmptyTable.AddData( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  query = np.array( [], dtype = np.int64 )
  result = EmptyTable.SameStart( query )
  assert result is None


def test_samestart_shorter_sequence_overwrites_longer( EmptyTable ) -> None:
  '''ORIGINAL: When a shorter sequence arrives, it overwrites the longer one.'''
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = EmptyTable.AddData( long_seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = long_seq, Value = idx_long )

  short_seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = EmptyTable.AddData( short_seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = short_seq, Value = idx_short )

  query = np.array( [ 2, 1 ], dtype = np.int64 )
  result = EmptyTable.SameStart( query )
  assert isinstance( result, int )
  assert result == idx_short


def test_samestart_with_non_existent_key_returns_none( EmptyTable ) -> None:
  '''SameStart with a non-existent prefix returns None.'''
  result = EmptyTable.SameStart( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  assert result is None


def test_samestart_permutation_invariance( EmptyTable ) -> None:
  '''SameStart is invariant to permutation of query elements.'''
  seq = np.array( [ 5, 1, 9 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  result1 = EmptyTable.SameStart( np.array( [ 1, 5 ], dtype = np.int64 ) )
  assert isinstance( result1, int )
  assert result1 == idx
  result2 = EmptyTable.SameStart( np.array( [ 5, 1 ], dtype = np.int64 ) )
  assert isinstance( result2, int )
  assert result2 == idx
  result3 = EmptyTable.SameStart( np.array( [ 1, 9 ], dtype = np.int64 ) )
  assert result3 is None


def test_empty_query_samestart( EmptyTable ) -> None:
  '''SameStart with an empty query array returns None.'''
  seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  result = EmptyTable.SameStart( np.array( [], dtype = np.int64 ) )
  assert result is None


# ----------------------------------------------------------------------
# Tests for CreateKeys
# ----------------------------------------------------------------------
def CheckKey( Table: MultiKeyHashTable, Query: NDArray, ExpectedIdx: int ) -> None:
  '''Helper: assert that SameStart for query returns expected_idx.'''
  result = Table.SameStart( Query )
  assert isinstance( result, int ), f"SameStart({ Query }) returned None, expected { ExpectedIdx }"
  assert result == ExpectedIdx, f"SameStart({ Query }) returned { result }, expected { ExpectedIdx }"


def CheckNoKey( Table: MultiKeyHashTable, Query: NDArray ) -> None:
  '''Helper: assert that SameStart for query returns None.'''
  assert Table.SameStart( Query ) is None, f"SameStart({ Query }) should have returned None"


def test_create_keys_generates_all_prefix_lengths( EmptyTable ) -> None:
  '''CreateKeys generates keys for all prefix lengths >= MinLen.'''
  seq = np.array( [ 3, 1, 4 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  CheckKey( EmptyTable, np.array( [ 3 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 1, 3 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 1, 3, 4 ], dtype = np.int64 ), idx )


def test_create_keys_minlen_zero_forced_to_one( EmptyTable ) -> None:
  '''MinLen=0 should be treated as 1 (no empty tuples).'''
  seq = np.array( [ 5, 2 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 0, IndexSet = seq, Value = idx )

  CheckKey( EmptyTable, np.array( [ 5 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 2, 5 ], dtype = np.int64 ), idx )


def test_create_keys_with_minlen_greater_than_one( EmptyTable ) -> None:
  '''CreateKeys with MinLen > 1 only generates keys of that length or greater.'''
  seq = np.array( [ 7, 8, 9, 10 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 3, IndexSet = seq, Value = idx )

  CheckNoKey( EmptyTable, np.array( [ 7 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 7, 8 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 7, 8, 9 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 7, 8, 9, 10 ], dtype = np.int64 ), idx )


def test_create_keys_duplicate_has_no_effect( EmptyTable ) -> None:
  '''Creating keys for the same sequence twice leaves the first mapping intact.'''
  seq = np.array( [ 5, 6 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  CheckKey( EmptyTable, np.array( [ 5, 6 ], dtype = np.int64 ), idx )

  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  CheckKey( EmptyTable, np.array( [ 5, 6 ], dtype = np.int64 ), idx )


def test_create_keys_does_not_overwrite_with_longer_sequence( EmptyTable ) -> None:
  '''First-inserted (shorter or longer) key is preserved.'''
  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = EmptyTable.AddData( short )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = short, Value = idx_short )

  long = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = EmptyTable.AddData( long )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = long, Value = idx_long )

  CheckKey( EmptyTable, np.array( [ 1, 2 ], dtype = np.int64 ), idx_short )


def test_create_keys_shorter_sequence_overwrites( EmptyTable ) -> None:
  '''ORIGINAL: longer-stored key is overwritten by shorter sequence.'''
  long = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = EmptyTable.AddData( long )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = long, Value = idx_long )

  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = EmptyTable.AddData( short )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = short, Value = idx_short )

  CheckKey( EmptyTable, np.array( [ 1, 2 ], dtype = np.int64 ), idx_short )
  CheckKey( EmptyTable, np.array( [ 1, 2, 3 ], dtype = np.int64 ), idx_long )


def test_create_keys_with_empty_indexset( EmptyTable ) -> None:
  '''CreateKeys with an empty IndexSet creates no keys.'''
  empty_arr = np.array( [], dtype = np.int64 )
  idx = EmptyTable.AddData( empty_arr )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = empty_arr, Value = idx )
  CheckNoKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ) )


def test_minlen_greater_than_indexset_length( EmptyTable ) -> None:
  '''MinLen greater than IndexSet length creates no keys.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 5, IndexSet = seq, Value = idx )
  CheckNoKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ) )


# ----------------------------------------------------------------------
# Tests for DeleteAllOfSize
# ----------------------------------------------------------------------
def test_delete_all_of_size_removes_exact_length( EmptyTable ) -> None:
  '''DeleteAllOfSize removes only keys of exactly length n.'''
  seq1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx1 = EmptyTable.AddData( seq1 )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq1, Value = idx1 )

  seq2 = np.array( [ 4, 5, 6, 7 ], dtype = np.int64 )
  idx2 = EmptyTable.AddData( seq2 )
  EmptyTable.CreateKeys( MinLen = 2, IndexSet = seq2, Value = idx2 )

  CheckKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ), idx1 )
  CheckKey( EmptyTable, np.array( [ 4, 5 ], dtype = np.int64 ), idx2 )

  EmptyTable.DeleteAllOfSize( 1 )
  CheckNoKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 4, 5 ], dtype = np.int64 ), idx2 )


def test_delete_all_of_size_with_n_equal_max_length( EmptyTable ) -> None:
  '''ORIGINAL: DeleteAllOfSize(n) deletes all keys with len <= n.'''
  seq = np.array( [ 10, 20 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 2 )
  CheckNoKey( EmptyTable, np.array( [ 10 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20 ], dtype = np.int64 ) )


def test_delete_all_of_size_raises_on_non_int( EmptyTable ) -> None:
  '''DeleteAllOfSize raises TypeError for non-integer n.'''
  with pytest.raises( TypeError, match = "n must be an int" ):
    EmptyTable.DeleteAllOfSize( "not an int" )
  with pytest.raises( TypeError, match = "n must be an int" ):
    EmptyTable.DeleteAllOfSize( 2.0 )


def test_delete_all_of_size_does_not_affect_data( TableInt64 ) -> None:
  '''DeleteAllOfSize only affects keys, not stored data.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = TableInt64.AddData( seq )
  TableInt64.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  TableInt64.DeleteAllOfSize( 1 )
  TableInt64.DeleteAllOfSize( 2 )
  TableInt64.DeleteAllOfSize( 3 )
  assert np.array_equal( TableInt64[ idx ], seq )


def test_delete_all_of_size_removes_up_to_n_inclusive( EmptyTable ) -> None:
  '''ORIGINAL: DeleteAllOfSize(n) removes all keys with len <= n.'''
  seq = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 2 )
  CheckNoKey( EmptyTable, np.array( [ 10 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 10, 20, 30 ], dtype = np.int64 ), idx )


def test_delete_all_of_size_with_large_n( EmptyTable ) -> None:
  '''ORIGINAL: DeleteAllOfSize(10) deletes all existing keys (all len <= 10).'''
  seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  CheckKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ), idx )
  EmptyTable.DeleteAllOfSize( 10 )
  CheckNoKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 1, 2 ], dtype = np.int64 ) )


def test_delete_all_of_size_with_non_positive_n( EmptyTable ) -> None:
  '''DeleteAllOfSize with n=0 or negative is a no-op (no zero-length keys).'''
  seq = np.array( [ 1 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  EmptyTable.DeleteAllOfSize( 0 )
  CheckKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ), idx )
  EmptyTable.DeleteAllOfSize( -1 )
  CheckKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ), idx )


# ----------------------------------------------------------------------
# Integration tests simulating arborescence usage
# ----------------------------------------------------------------------
def test_typical_arborescence_flow( EmptyTable ) -> None:
  '''Simulate a simplified BFS where LI increases each level.'''
  li_level1 = np.array( [ 0 ], dtype = np.int64 )
  ellg_level1 = np.array( [ 0, 3, 7 ], dtype = np.int64 )
  idx1 = EmptyTable.AddData( ellg_level1 )
  EmptyTable.CreateKeys( MinLen = len( li_level1 ), IndexSet = ellg_level1, Value = idx1 )

  CheckKey( EmptyTable, np.array( [ 0 ], dtype = np.int64 ), idx1 )

  li_level2 = np.array( [ 0, 3 ], dtype = np.int64 )
  CheckKey( EmptyTable, li_level2, idx1 )

  EmptyTable.DeleteAllOfSize( 1 )
  CheckNoKey( EmptyTable, np.array( [ 0 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 0, 3 ], dtype = np.int64 ), idx1 )

  ellg_level2 = np.array( [ 0, 3, 5, 9 ], dtype = np.int64 )
  idx2 = EmptyTable.AddData( ellg_level2 )
  EmptyTable.CreateKeys( MinLen = len( li_level2 ), IndexSet = ellg_level2, Value = idx2 )

  CheckKey( EmptyTable, np.array( [ 0, 3 ], dtype = np.int64 ), idx1 )

  EmptyTable.DeleteAllOfSize( 2 )
  CheckNoKey( EmptyTable, np.array( [ 0, 3 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 0, 3, 5 ], dtype = np.int64 ), idx2 )


def test_random_arborescence_consistency( EmptyTable ) -> None:
  '''Simulate a mini arborescence: incremental LI, add regressions, query.'''
  rng = np.random.default_rng( 42 )
  for level_size in range( 1, 6 ):
    LI = rng.choice( 10, size = level_size, replace = False ).astype( np.int64 )
    hit = EmptyTable.SameStart( LI )
    if ( hit is not None ):
      assert isinstance( hit, int )
      stored = EmptyTable[ hit ]
      assert np.array_equal( np.sort( stored[ : level_size ] ), np.sort( LI ) )

    extra = rng.choice( 10, size = rng.integers( 0, 3 ), replace = False )
    full = np.concatenate( [ LI, extra ] )
    idx = EmptyTable.AddData( full )
    EmptyTable.CreateKeys( MinLen = len( LI ), IndexSet = full, Value = idx )

    EmptyTable.DeleteAllOfSize( level_size )

  for key_len in range( 1, len( EmptyTable.Tables ) ):
    if ( EmptyTable.Tables[ key_len ] is not None ): assert key_len > 5


# ----------------------------------------------------------------------
# Performance / stress
# ----------------------------------------------------------------------
def test_large_number_of_keys_stress( EmptyTable ) -> None:
  '''Simple stress test: create many keys and check retrieval.'''
  for i in range( 100 ):
    seq = np.arange( i, i + 5, dtype = np.int64 )
    idx = EmptyTable.AddData( seq )
    EmptyTable.CreateKeys( MinLen = 3, IndexSet = seq, Value = idx )

  for i in range( 0, 100, 10 ):
    seq = np.arange( i, i + 5, dtype = np.int64 )
    query = np.sort( seq[ : 3 ] )
    result = EmptyTable.SameStart( query )
    assert isinstance( result, int )
    assert result == i


# ----------------------------------------------------------------------
# Tests for __len__
# ----------------------------------------------------------------------
def test_len_empty( EmptyTable: MultiKeyHashTable ) -> None:
  '''An empty table has length 0.'''
  assert len( EmptyTable ) == 0


def test_len_after_add( TableInt64: MultiKeyHashTable ) -> None:
  '''Length grows correctly after each AddData call.'''
  assert len( TableInt64 ) == 0
  TableInt64.AddData( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  assert len( TableInt64 ) == 1
  TableInt64.AddData( np.array( [ 4, 5 ], dtype = np.int64 ) )
  assert len( TableInt64 ) == 2
  TableInt64.AddData( np.array( [], dtype = np.int64 ) )
  assert len( TableInt64 ) == 3


# ----------------------------------------------------------------------
# Tests for __iter__
# ----------------------------------------------------------------------
def test_iter_empty( EmptyTable: MultiKeyHashTable ) -> None:
  '''Iterating over an empty table yields nothing.'''
  assert list( EmptyTable ) == []


def test_iter_yields_stored_sequences( TableInt64: MultiKeyHashTable ) -> None:
  '''Iterating yields each stored sequence in the order they were added.'''
  a1 = np.array( [ 3, 1, 5 ], dtype = np.int64 )
  a2 = np.array( [ 2, 4 ], dtype = np.int64 )
  a3 = np.array( [ 10 ], dtype = np.int64 )
  TableInt64.AddData( a1 )
  TableInt64.AddData( a2 )
  TableInt64.AddData( a3 )
  results = list( TableInt64 )
  assert len( results ) == 3
  assert np.array_equal( results[ 0 ], a1 )
  assert results[ 0 ].dtype == np.int64
  assert np.array_equal( results[ 1 ], a2 )
  assert results[ 1 ].dtype == np.int64
  assert np.array_equal( results[ 2 ], a3 )
  assert results[ 2 ].dtype == np.int64


def test_iter_matches_getitem( TableInt64: MultiKeyHashTable ) -> None:
  '''For every i, the i-th element from iteration equals table[i].'''
  TableInt64.AddData( np.array( [ 7, 2, 9 ], dtype = np.int64 ) )
  TableInt64.AddData( np.array( [ 5, 1 ], dtype = np.int64 ) )
  TableInt64.AddData( np.array( [ 42 ], dtype = np.int64 ) )
  for i, seq in enumerate( TableInt64 ):
    assert np.array_equal( seq, TableInt64[ i ] )
    assert seq.dtype == np.int64


# ----------------------------------------------------------------------
# Regression tests — verify ORIGINAL (dict-based) behavior against NEW
# ----------------------------------------------------------------------
def test_create_keys_overwrites_for_shorter_sequence( EmptyTable: MultiKeyHashTable ) -> None:
  '''ORIGINAL overwrites key when stored sequence is LONGER than IndexSet
  (shorter sequence wins). Both share same unsorted prefix, so keys collide.
  NEW: fingerprint match keep first (long). ORIGINAL: overwrite with short.'''
  longer = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = EmptyTable.AddData( longer )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = longer, Value = idx_long )

  shorter = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = EmptyTable.AddData( shorter )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = shorter, Value = idx_short )

  CheckKey( EmptyTable, np.array( [ 1, 2 ], dtype = np.int64 ), idx_short )


def test_delete_all_of_size_removes_up_to_n( EmptyTable: MultiKeyHashTable ) -> None:
  '''ORIGINAL DeleteAllOfSize(n) deletes keys with len(reg) <= n.
  NEW deletes only exact len == n. This test verifies ORIGINAL's <= semantics.'''
  seq = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 2 )
  CheckNoKey( EmptyTable, np.array( [ 10 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 10, 20, 30 ], dtype = np.int64 ), idx )


def test_create_keys_overwrite_three_prefixes_of_different_lengths( EmptyTable: MultiKeyHashTable ) -> None:
  '''ORIGINAL overwrites each shared prefix when a shorter sequence arrives.
  Both share same prefix elements [1,2], so level-1 and level-2 keys collide.
  ORIGINAL overwrites both to short; level-3 key (longer only) stays on long.'''
  longer = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = EmptyTable.AddData( longer )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = longer, Value = idx_long )

  shorter = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = EmptyTable.AddData( shorter )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = shorter, Value = idx_short )

  CheckKey( EmptyTable, np.array( [ 1 ], dtype = np.int64 ), idx_short )
  CheckKey( EmptyTable, np.array( [ 1, 2 ], dtype = np.int64 ), idx_short )
  CheckKey( EmptyTable, np.array( [ 1, 2, 3 ], dtype = np.int64 ), idx_long )


def test_samestart_returns_correct_data_after_insert( EmptyTable: MultiKeyHashTable ) -> None:
  '''SameStart must return an index whose stored data actually matches the query
  (catches missing exact-verification in NEW).'''
  rng = np.random.default_rng( 42 )
  for _ in range( 100 ):
    n = int( rng.integers( 1, 8 ) )
    seq = rng.choice( 50, size = n, replace = False ).astype( np.int64 )
    idx = EmptyTable.AddData( seq )
    EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

    for k in range( 1, n + 1 ):
      prefix = seq[ : k ]
      result = EmptyTable.SameStart( np.sort( prefix ) )
      assert result is not None, f"SameStart({ prefix }) returned None for seq={ seq }"
      stored = EmptyTable[ result ]
      assert np.array_equal( np.sort( stored[ : k ] ), np.sort( prefix ) ), f"Index { result } stored={ stored } does not match query prefix={ prefix } from seq={ seq }"


def test_delete_all_of_size_zero_and_negative( EmptyTable: MultiKeyHashTable ) -> None:
  '''DeleteAllOfSize(0): ORIGINAL deletes len-0 keys; NEW no-ops since no len-0 keys.
  DeleteAllOfSize(-1): both are no-ops. Both should leave len-1+ keys intact.'''
  seq = np.array( [ 5, 6, 7 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 0 )
  CheckKey( EmptyTable, np.array( [ 5 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 5, 6, 7 ], dtype = np.int64 ), idx )

  EmptyTable.DeleteAllOfSize( -1 )
  CheckKey( EmptyTable, np.array( [ 5 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 5, 6 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 5, 6, 7 ], dtype = np.int64 ), idx )


def test_delete_all_of_size_kills_lengths_up_to_n( EmptyTable: MultiKeyHashTable ) -> None:
  '''ORIGINAL <= semantics: DeleteAllOfSize(2) kills length-1 and length-2 keys.
  NEW == semantics: kills only length 2, leaves length 1 alive.'''
  seq = np.array( [ 10, 20, 30, 40 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 2 )
  CheckNoKey( EmptyTable, np.array( [ 10 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 10, 20, 30 ], dtype = np.int64 ), idx )
  CheckKey( EmptyTable, np.array( [ 10, 20, 30, 40 ], dtype = np.int64 ), idx )


def test_getitem_returns_copy( EmptyTable: MultiKeyHashTable ) -> None:
  '''__getitem__ returns a copy; mutating the result does not corrupt storage.'''
  original = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = EmptyTable.AddData( original )

  retrieved = EmptyTable[ idx ]
  retrieved[ 0 ] = 99

  assert np.array_equal( EmptyTable[ idx ], original )


def test_add_data_copies_input( EmptyTable: MultiKeyHashTable ) -> None:
  '''AddData copies; modifying the source after AddData is safe.'''
  source = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  idx = EmptyTable.AddData( source )

  source[ 0 ] = 999
  source[ 1 ] = 888

  stored = EmptyTable[ idx ]
  assert np.array_equal( stored, np.array( [ 10, 20, 30 ], dtype = np.int64 ) )


def test_lengths_of_stored_sequences_match( EmptyTable: MultiKeyHashTable ) -> None:
  '''Each retrieved sequence has the correct length matching what was stored.'''
  lengths = [ 0, 1, 5, 2, 3, 10, 0 ]
  for L in lengths:
    arr = np.arange( L, dtype = np.int64 )
    EmptyTable.AddData( arr )

  for i, expected_len in enumerate( lengths ):
    stored = EmptyTable[ i ]
    assert len( stored ) == expected_len, f"Index { i }: expected length { expected_len }, got { len( stored ) }"


def test_samestart_verifies_all_prefixes_after_many_inserts( EmptyTable: MultiKeyHashTable ) -> None:
  '''Every prefix of every inserted sequence must be findable via SameStart and
  must point to data whose first k elements match the query.'''
  rng = np.random.default_rng( 12345 )
  indices: list[ int ] = []
  for _ in range( 30 ):
    n = int( rng.integers( 2, 7 ) )
    seq = rng.choice( 200, size = n, replace = False ).astype( np.int64 )
    idx = EmptyTable.AddData( seq )
    EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
    indices.append( idx )

  for _ in range( 200 ):
    k = int( rng.integers( 1, 7 ) )
    query = rng.choice( 200, size = k, replace = False ).astype( np.int64 )
    result = EmptyTable.SameStart( query )
    if ( result is not None ):
      stored = EmptyTable[ result ]
      assert np.array_equal( np.sort( stored[ : k ] ), np.sort( query ) ), f"Stored={ stored } at index { result } does not match query={ query }"


def test_delete_all_of_size_edge_order( EmptyTable: MultiKeyHashTable ) -> None:
  '''DeleteAllOfSize called twice with <= semantics still leaves longer keys.'''
  seq = np.array( [ 10, 20, 30, 40 ], dtype = np.int64 )
  idx = EmptyTable.AddData( seq )
  EmptyTable.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  EmptyTable.DeleteAllOfSize( 2 )
  EmptyTable.DeleteAllOfSize( 3 )

  CheckNoKey( EmptyTable, np.array( [ 10 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20 ], dtype = np.int64 ) )
  CheckNoKey( EmptyTable, np.array( [ 10, 20, 30 ], dtype = np.int64 ) )
  CheckKey( EmptyTable, np.array( [ 10, 20, 30, 40 ], dtype = np.int64 ), idx )


# ======================================================================
# Freeze tests (converted from test_MultiKeyHashTable_freeze.py)
# Verify the implementation produces expected outputs.
# ======================================================================

def test_InitIdentical() -> None:
  tbl = MultiKeyHashTable()
  assert tbl.DataCount == 0
  assert list( tbl ) == []


def test_AddDataSingle() -> None:
  tbl = MultiKeyHashTable()
  arr = np.array( [ 3, 1, 4 ], dtype = np.int64 )
  idx = tbl.AddData( arr )
  assert idx == 0
  assert np.array_equal( tbl[ 0 ], arr )


def test_AddDataSequential() -> None:
  tbl = MultiKeyHashTable()
  arrays = [
        np.array( [ 5, 2, 9 ], dtype = np.int64 ),
        np.array( [ 1, 7 ], dtype = np.int64 ),
        np.array( [ 42 ], dtype = np.int64 ),
    ]
  indices = [ tbl.AddData( arr ) for arr in arrays ]
  assert indices == [ 0, 1, 2 ]
  for i in range( len( arrays ) ): assert np.array_equal( tbl[ i ], arrays[ i ] )


def test_AddDataEmptyArray() -> None:
  tbl = MultiKeyHashTable()
  empty_arr = np.array( [], dtype = np.int64 )
  idx = tbl.AddData( empty_arr )
  assert idx == 0
  assert np.array_equal( tbl[ 0 ], empty_arr )


def test_AddDataCopiesInput() -> None:
  tbl = MultiKeyHashTable()
  source = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  idx = tbl.AddData( source )
  source[ 0 ] = 999
  source[ 1 ] = 888
  expected = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  assert np.array_equal( tbl[ idx ], expected )


def test_AddDataTwiceSameArray() -> None:
  tbl = MultiKeyHashTable()
  arr = np.array( [ 7, 8 ], dtype = np.int64 )
  idx1 = tbl.AddData( arr )
  idx2 = tbl.AddData( arr )
  assert idx1 == 0
  assert idx2 == 1
  assert idx2 == idx1 + 1
  assert np.array_equal( tbl[ 0 ], arr )
  assert np.array_equal( tbl[ 1 ], arr )


def test_GetItemPositive() -> None:
  tbl = MultiKeyHashTable()
  a1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  a2 = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  idx_a = tbl.AddData( a1 )
  tbl.AddData( a2 )
  assert np.array_equal( tbl[ idx_a ], a1 )
  assert np.array_equal( tbl[ 1 ], a2 )


def test_GetItemNegative() -> None:
  tbl = MultiKeyHashTable()
  a1 = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  a2 = np.array( [ 40, 50 ], dtype = np.int64 )
  tbl.AddData( a1 )
  tbl.AddData( a2 )
  assert np.array_equal( tbl[ -1 ], a2 )
  assert np.array_equal( tbl[ -2 ], a1 )


def test_GetItemRaisesIndexError() -> None:
  tbl = MultiKeyHashTable()
  with pytest.raises( IndexError ):
    _ = tbl[ 0 ]
  with pytest.raises( IndexError ):
    _ = tbl[ -1 ]
  arr = np.array( [ 1 ], dtype = np.int64 )
  tbl.AddData( arr )
  with pytest.raises( IndexError ):
    _ = tbl[ 1 ]
  with pytest.raises( IndexError ):
    _ = tbl[ -2 ]


def test_GetItemAfterKeyDeletion() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 3 )
  assert np.array_equal( tbl[ idx ], seq )


def test_SameStartExactMatch() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 3, 1, 5, 7 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  query = np.array( [ 5, 1, 3 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result == 0


def test_SameStartSubsetMatch() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 3, 1, 5, 7 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  query = np.array( [ 1, 3 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result == 0


def test_SameStartNoMatch() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  query = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result is None


def test_SameStartEmptyQuery() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  query = np.array( [], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result is None


def test_SameStartNonExistentKey() -> None:
  tbl = MultiKeyHashTable()
  query = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result is None


def test_SameStartPermutationInvariance() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 5, 1, 9 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  assert tbl.SameStart( np.array( [ 1, 5 ], dtype = np.int64 ) ) == idx
  assert tbl.SameStart( np.array( [ 5, 1 ], dtype = np.int64 ) ) == idx
  assert tbl.SameStart( np.array( [ 1, 9 ], dtype = np.int64 ) ) is None


def test_SameStartAfterCreateKeysAllPrefixes() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 3, 1, 4, 1, 5 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  for i in range( 1, 6 ):
    prefix = seq[ : i ]
    query = np.sort( prefix )
    result = tbl.SameStart( query )
    assert result == idx


def test_SameStartShorterOverwritesLonger() -> None:
  tbl = MultiKeyHashTable()
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = tbl.AddData( long_seq )
  tbl.CreateKeys( 1, long_seq, idx_long )
  short_seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = tbl.AddData( short_seq )
  tbl.CreateKeys( 1, short_seq, idx_short )
  query = np.array( [ 2, 1 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result == idx_short


def test_SameStartDoesNotOverwriteWhenEqualLen() -> None:
  tbl = MultiKeyHashTable()
  seq1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx1 = tbl.AddData( seq1 )
  tbl.CreateKeys( 1, seq1, idx1 )
  seq2 = np.array( [ 1, 2, 4 ], dtype = np.int64 )
  idx2 = tbl.AddData( seq2 )
  tbl.CreateKeys( 1, seq2, idx2 )
  query = np.array( [ 1, 2 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result == idx1


def test_SameStartDataMatchesQuery() -> None:
  tbl = MultiKeyHashTable()
  rng = np.random.default_rng( 42 )
  for _ in range( 50 ):
    n = int( rng.integers( 1, 8 ) )
    seq = rng.choice( 50, size = n, replace = False ).astype( np.int64 )
    idx = tbl.AddData( seq )
    tbl.CreateKeys( 1, seq, idx )
    for k in range( 1, n + 1 ):
      prefix = seq[ : k ]
      query = np.sort( prefix )
      result = tbl.SameStart( query )
      if ( result is not None ):
        stored = tbl[ result ]
        if ( not np.array_equal( np.sort( stored[ : k ] ), np.sort( prefix ) ) ):
          raise AssertionError(
                        f"Index { result } stored={ stored } does not match query prefix={ prefix }" )


def test_CreateKeysAllPrefixLengths() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 3, 1, 4 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  for i in range( 1, 4 ):
    prefix = seq[ : i ]
    query = np.sort( prefix )
    result = tbl.SameStart( query )
    assert result == idx


def test_CreateKeysMinLenZeroForcedToOne() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 5, 2 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 0, seq, idx )
  for i in range( 1, 3 ):
    query = np.sort( seq[ : i ] )
    result = tbl.SameStart( query )
    assert result == idx
  empty_query = np.array( [], dtype = np.int64 )
  assert tbl.SameStart( empty_query ) is None


def test_CreateKeysMinLenGreaterThanOne() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 7, 8, 9, 10 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 3, seq, idx )
  for i in range( 1, 3 ):
    query = np.sort( seq[ : i ] )
    assert tbl.SameStart( query ) is None
  for i in range( 3, 5 ):
    query = np.sort( seq[ : i ] )
    assert tbl.SameStart( query ) == idx


def test_CreateKeysEmptyIndexSet() -> None:
  tbl = MultiKeyHashTable()
  empty_arr = np.array( [], dtype = np.int64 )
  idx = tbl.AddData( empty_arr )
  tbl.CreateKeys( 1, empty_arr, idx )
  query = np.array( [ 1 ], dtype = np.int64 )
  assert tbl.SameStart( query ) is None


def test_CreateKeysMinLenGreaterThanIndexSet() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 5, seq, idx )
  for i in range( 1, 4 ):
    query = np.sort( seq[ : i ] )
    assert tbl.SameStart( query ) is None


def test_CreateKeysDuplicateCall() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 5, 6 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.CreateKeys( 1, seq, idx )
  query = np.sort( seq )
  assert tbl.SameStart( query ) == idx


def test_CreateKeysLongerDoesNotOverwriteShorter() -> None:
  tbl = MultiKeyHashTable()
  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = tbl.AddData( short )
  tbl.CreateKeys( 1, short, idx_short )
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = tbl.AddData( long_seq )
  tbl.CreateKeys( 1, long_seq, idx_long )
  query = np.array( [ 1, 2 ], dtype = np.int64 )
  result = tbl.SameStart( query )
  assert result == idx_short


def test_CreateKeysShorterOverwritesLonger() -> None:
  tbl = MultiKeyHashTable()
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = tbl.AddData( long_seq )
  tbl.CreateKeys( 1, long_seq, idx_long )
  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = tbl.AddData( short )
  tbl.CreateKeys( 1, short, idx_short )
  query12 = np.array( [ 1, 2 ], dtype = np.int64 )
  assert tbl.SameStart( query12 ) == idx_short
  query123 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  assert tbl.SameStart( query123 ) == idx_long


def test_CreateKeysThreePrefixesOverwrite() -> None:
  tbl = MultiKeyHashTable()
  longer = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = tbl.AddData( longer )
  tbl.CreateKeys( 1, longer, idx_long )
  shorter = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = tbl.AddData( shorter )
  tbl.CreateKeys( 1, shorter, idx_short )
  q1 = np.array( [ 1 ], dtype = np.int64 )
  assert tbl.SameStart( q1 ) == idx_short
  q2 = np.array( [ 1, 2 ], dtype = np.int64 )
  assert tbl.SameStart( q2 ) == idx_short
  q3 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  assert tbl.SameStart( q3 ) == idx_long


def test_DeleteAllOfSizeRemoveUpToN() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 1 )
  assert tbl.SameStart( np.array( [ 1 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 1, 2 ], dtype = np.int64 ) ) == idx
  assert tbl.SameStart( np.array( [ 1, 2, 3 ], dtype = np.int64 ) ) == idx


def test_DeleteAllOfSizeRemoveUpToNInclusive() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 10, 20, 30, 40 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 2 )
  assert tbl.SameStart( np.array( [ 10 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 10, 20 ], dtype = np.int64 ) ) is None
  r3 = tbl.SameStart( np.array( [ 10, 20, 30 ], dtype = np.int64 ) )
  assert r3 == idx
  r4 = tbl.SameStart( np.array( [ 10, 20, 30, 40 ], dtype = np.int64 ) )
  assert r4 == idx


def test_DeleteAllOfSizeLargeN() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 10 )
  assert tbl.SameStart( np.array( [ 1 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 1, 2 ], dtype = np.int64 ) ) is None


def test_DeleteAllOfSizeMultipleCalls() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 10, 20, 30, 40 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 2 )
  tbl.DeleteAllOfSize( 3 )
  assert tbl.SameStart( np.array( [ 10 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 10, 20 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 10, 20, 30 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( np.array( [ 10, 20, 30, 40 ], dtype = np.int64 ) ) == idx


def test_DeleteAllOfSizeZeroAndNegative() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 5, 6, 7 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 0 )
  assert tbl.SameStart( np.array( [ 5 ], dtype = np.int64 ) ) == idx
  assert tbl.SameStart( np.array( [ 5, 6, 7 ], dtype = np.int64 ) ) == idx
  tbl.DeleteAllOfSize( -1 )
  assert tbl.SameStart( np.array( [ 5 ], dtype = np.int64 ) ) == idx


def test_DeleteAllOfSizeNonIntRaises() -> None:
  tbl = MultiKeyHashTable()
  with pytest.raises( ( TypeError ) ):
    tbl.DeleteAllOfSize( "not an int" )
  with pytest.raises( ( TypeError ) ):
    tbl.DeleteAllOfSize( 2.0 )


def test_DeleteAllOfSizeDataNotAffected() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  tbl.DeleteAllOfSize( 1 )
  tbl.DeleteAllOfSize( 2 )
  tbl.DeleteAllOfSize( 3 )
  assert np.array_equal( tbl[ idx ], seq )


def test_DeleteAllOfSizePartialDataPreserved() -> None:
  tbl = MultiKeyHashTable()
  seq1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  seq2 = np.array( [ 4, 5, 6, 7 ], dtype = np.int64 )
  idx1 = tbl.AddData( seq1 )
  idx2 = tbl.AddData( seq2 )
  tbl.CreateKeys( 1, seq1, idx1 )
  tbl.CreateKeys( 2, seq2, idx2 )
  tbl.DeleteAllOfSize( 1 )
  assert tbl.SameStart( np.array( [ 1 ], dtype = np.int64 ) ) is None
  r2 = tbl.SameStart( np.array( [ 4, 5 ], dtype = np.int64 ) )
  assert r2 == idx2


def test_LenEmpty() -> None:
  tbl = MultiKeyHashTable()
  assert len( tbl ) == 0


def test_LenAfterAdd() -> None:
  tbl = MultiKeyHashTable()
  arr1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  arr2 = np.array( [ 4, 5 ], dtype = np.int64 )
  arr3 = np.array( [], dtype = np.int64 )
  tbl.AddData( arr1 )
  assert len( tbl ) == 1
  tbl.AddData( arr2 )
  assert len( tbl ) == 2
  tbl.AddData( arr3 )
  assert len( tbl ) == 3


def test_LenAfterDeleteAllOfSize() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, 0 )
  tbl.DeleteAllOfSize( 3 )
  assert len( tbl ) == 1


def test_IterEmpty() -> None:
  tbl = MultiKeyHashTable()
  assert list( tbl ) == []


def test_IterYieldsStoredSequences() -> None:
  tbl = MultiKeyHashTable()
  a1 = np.array( [ 3, 1, 5 ], dtype = np.int64 )
  a2 = np.array( [ 2, 4 ], dtype = np.int64 )
  a3 = np.array( [ 10 ], dtype = np.int64 )
  tbl.AddData( a1 )
  tbl.AddData( a2 )
  tbl.AddData( a3 )
  results = list( tbl )
  assert len( results ) == 3
  assert np.array_equal( results[ 0 ], a1 )
  assert np.array_equal( results[ 1 ], a2 )
  assert np.array_equal( results[ 2 ], a3 )


def test_IterMatchesGetItem() -> None:
  tbl = MultiKeyHashTable()
  tbl.AddData( np.array( [ 7, 2, 9 ], dtype = np.int64 ) )
  tbl.AddData( np.array( [ 5, 1 ], dtype = np.int64 ) )
  for i, seq in enumerate( tbl ): assert np.array_equal( seq, tbl[ i ] )


def test_GetMemoryUsageReturnsInt() -> None:
  tbl = MultiKeyHashTable()
  seq = np.array( [ 1, 2, 3, 4, 5 ], dtype = np.int64 )
  idx = tbl.AddData( seq )
  tbl.CreateKeys( 1, seq, idx )
  assert isinstance( tbl.get_memory_usage(), dict )


def test_IntegrationArborescenceFlow() -> None:
  tbl = MultiKeyHashTable()
  li1 = np.array( [ 0 ], dtype = np.int64 )
  ellg1 = np.array( [ 0, 3, 7 ], dtype = np.int64 )
  idx1 = tbl.AddData( ellg1 )
  tbl.CreateKeys( len( li1 ), ellg1, idx1 )
  assert tbl.SameStart( np.array( [ 0 ], dtype = np.int64 ) ) == idx1
  li2 = np.array( [ 0, 3 ], dtype = np.int64 )
  assert tbl.SameStart( li2 ) == idx1
  tbl.DeleteAllOfSize( 1 )
  assert tbl.SameStart( np.array( [ 0 ], dtype = np.int64 ) ) is None
  assert tbl.SameStart( li2 ) == idx1
  ellg2 = np.array( [ 0, 3, 5, 9 ], dtype = np.int64 )
  idx2 = tbl.AddData( ellg2 )
  tbl.CreateKeys( len( li2 ), ellg2, idx2 )
  assert tbl.SameStart( li2 ) == idx1
  tbl.DeleteAllOfSize( 2 )
  assert tbl.SameStart( li2 ) is None
  assert tbl.SameStart( np.array( [ 0, 3, 5 ], dtype = np.int64 ) ) == idx2


def test_IntegrationRandomConsistency() -> None:
  tbl = MultiKeyHashTable()
  rng = np.random.default_rng( 12345 )
  for _ in range( 30 ):
    n = int( rng.integers( 2, 7 ) )
    seq = rng.choice( 200, size = n, replace = False ).astype( np.int64 )
    idx = tbl.AddData( seq )
    tbl.CreateKeys( 1, seq, idx )
  for _ in range( 100 ):
    k = int( rng.integers( 1, 7 ) )
    query = rng.choice( 200, size = k, replace = False ).astype( np.int64 )
    result = tbl.SameStart( query )
    if ( result is not None ):
      stored = tbl[ result ]
      if ( not np.array_equal( np.sort( stored[ : k ] ), np.sort( query ) ) ):
        raise AssertionError(
                    f"Stored={ stored } at index { result } does not match query={ query }" )


def test_IntegrationRandomSequenceBFS() -> None:
  tbl = MultiKeyHashTable()
  rng = np.random.default_rng( 42 )
  for level_size in range( 1, 6 ):
    LI = rng.choice( 10, size = level_size, replace = False ).astype( np.int64 )
    hit = tbl.SameStart( LI )
    if ( hit is not None ):
      stored = tbl[ hit ]
      if ( not np.array_equal( np.sort( stored[ : level_size ] ), np.sort( LI ) ) ):
        raise AssertionError(
                    f"Stored={ stored } at index { hit } does not match LI={ LI }" )
    extra = rng.choice( 10, size = int( rng.integers( 0, 3 ) ), replace = False )
    full = np.concatenate( [ LI, extra ] ).astype( np.int64 )
    idx = tbl.AddData( full )
    tbl.CreateKeys( len( LI ), full, idx )
    tbl.DeleteAllOfSize( level_size )
  for level_size in range( 1, 6 ):
    for i in range( 1, level_size + 1 ):
      LI = rng.choice( 10, size = i, replace = False ).astype( np.int64 )
      _ = tbl.SameStart( LI )


def test_RegressionDifferentOrderingSamePrefix() -> None:
  tbl = MultiKeyHashTable()
  long_seq = np.array( [ 3, 1, 5 ], dtype = np.int64 )
  idx_long = tbl.AddData( long_seq )
  tbl.CreateKeys( 1, long_seq, idx_long )
  short_seq = np.array( [ 1, 3 ], dtype = np.int64 )
  idx_short = tbl.AddData( short_seq )
  tbl.CreateKeys( 1, short_seq, idx_short )
  q = np.array( [ 1, 3 ], dtype = np.int64 )
  result = tbl.SameStart( q )
  assert result == idx_short


def test_RegressionDifferentOrderingLongerPrefix() -> None:
  tbl = MultiKeyHashTable()
  seq_a = np.array( [ 5, 3, 7 ], dtype = np.int64 )
  idx_a = tbl.AddData( seq_a )
  tbl.CreateKeys( 1, seq_a, idx_a )
  seq_b = np.array( [ 3, 5 ], dtype = np.int64 )
  idx_b = tbl.AddData( seq_b )
  tbl.CreateKeys( 1, seq_b, idx_b )
  q = np.array( [ 3, 5 ], dtype = np.int64 )
  assert tbl.SameStart( q ) == idx_b
  q2 = np.array( [ 3, 5, 7 ], dtype = np.int64 )
  assert tbl.SameStart( q2 ) == idx_a


def test_RegressionSameStartVerifiesManyInserts() -> None:
  tbl = MultiKeyHashTable()
  rng = np.random.default_rng( 99 )
  for _ in range( 50 ):
    n = int( rng.integers( 1, 8 ) )
    seq = rng.choice( 100, size = n, replace = False ).astype( np.int64 )
    idx = tbl.AddData( seq )
    tbl.CreateKeys( 1, seq, idx )
    for k in range( 1, n + 1 ):
      prefix = seq[ : k ]
      query = np.sort( prefix )
      result = tbl.SameStart( query )
      if ( result is not None ):
        stored = tbl[ result ]
        if ( not np.array_equal( np.sort( stored[ : k ] ), np.sort( prefix ) ) ):
          raise AssertionError(
                        f"Index { result } stored={ stored } does not match query prefix={ prefix }" )
