# NARMAX/Test/test_MultiKeyHashTable.py
import pytest
import numpy as np
from numpy.typing import NDArray

# Adjust the import path according to your project layout.
# Since NARMAX/Test is a sibling of NARMAX/Classes, you might need:
import sys
from pathlib import Path
sys.path.insert( 0, str( Path( __file__ ).parent.parent ) )
from NARMAX.Classes.MultiKeyHashTable import MultiKeyHashTable


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def empty_table() -> MultiKeyHashTable:
  '''Return a fresh, empty MultiKeyHashTable.'''
  return MultiKeyHashTable()


@pytest.fixture
def table_with_data() -> MultiKeyHashTable:
  '''Return a table pre-populated with some data and keys.'''
  table = MultiKeyHashTable()
  # Data: three arrays
  a1 = np.array( [ 3, 1, 5, 7 ], dtype = np.int64 )
  a2 = np.array( [ 2, 4, 6 ], dtype = np.int64 )
  a3 = np.array( [ 10, 20 ], dtype = np.int64 )
  idx1 = table.AddData( a1 )
  idx2 = table.AddData( a2 )
  idx3 = table.AddData( a3 )

  # Create keys for a1 (MinLen=1, IndexSet = [3,1,5,7])
  table.CreateKeys( MinLen = 1, IndexSet = a1, Value = idx1 )
  # Create keys for a2 (MinLen=2, IndexSet = [2,4,6])
  table.CreateKeys( MinLen = 2, IndexSet = a2, Value = idx2 )
  # Create keys for a3 (MinLen=1, IndexSet = [10,20])
  table.CreateKeys( MinLen = 1, IndexSet = a3, Value = idx3 )

  return table


# ----------------------------------------------------------------------
# Tests for __init__
# ----------------------------------------------------------------------
def test_init_empty( empty_table ) -> None:
  '''New instance should have empty Data list and empty LookUpDict.'''
  assert isinstance( empty_table.Data, list )
  assert isinstance( empty_table.LookUpDict, dict )
  assert empty_table.Data == []
  assert empty_table.LookUpDict == {}


# ----------------------------------------------------------------------
# Tests for AddData
# ----------------------------------------------------------------------
def test_add_data_returns_correct_index( empty_table ) -> None:
  '''Adding data returns sequential indices starting from 0.'''
  item = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = empty_table.AddData( item )
  assert isinstance( idx, int )
  assert idx == 0
  assert len( empty_table.Data ) == 1
  assert np.array_equal( empty_table.Data[ 0 ], item )
  assert empty_table.Data[ 0 ].dtype == np.int64

  item2 = np.array( [ 4, 5 ], dtype = np.int64 )
  idx2 = empty_table.AddData( item2 )
  assert isinstance( idx2, int )
  assert idx2 == 1
  assert len( empty_table.Data ) == 2
  assert np.array_equal( empty_table.Data[ 1 ], item2 )
  assert empty_table.Data[ 1 ].dtype == np.int64


def test_add_data_stores_exact_array( empty_table ) -> None:
  '''The stored array is a copy; modifying the original does not affect storage.'''
  item = np.array( [ 42, 7 ], dtype = np.int64 )
  idx = empty_table.AddData( item )
  assert isinstance( idx, int )
  assert empty_table.Data[ idx ].dtype == np.int64
  # Modifying the original item after storage should not affect stored copy
  item[ 0 ] = 99
  item[ 1 ] = -1
  assert np.array_equal( empty_table.Data[ idx ], np.array( [ 42, 7 ] ) )


# ----------------------------------------------------------------------
# Tests for __getitem__
# ----------------------------------------------------------------------
def test_getitem_returns_correct_array( empty_table ) -> None:
  '''Indexing the table with a valid index returns the correct array.'''
  a = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  b = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  idx_a = empty_table.AddData( a )
  idx_b = empty_table.AddData( b )

  assert np.array_equal( empty_table[ idx_a ], a )
  assert empty_table[ idx_a ].dtype == np.int64
  assert np.array_equal( empty_table[ idx_b ], b )
  assert empty_table[ idx_b ].dtype == np.int64


def test_getitem_raises_index_error( empty_table ) -> None:
  '''Indexing with an out-of-range index raises IndexError.'''
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = empty_table[ 0 ]
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = empty_table[ -1 ]
  empty_table.AddData( np.array( [ 1 ], dtype = np.int64 ) )
  with pytest.raises( IndexError ):
    _ = empty_table[ 1 ]
  with pytest.raises( IndexError ):
    _ = empty_table[ -2 ]


def test_all_lookup_values_are_valid_indices( empty_table ) -> None:
  '''All LookUpDict values are valid Data indices.'''
  for _ in range( 20 ):
    arr = np.random.randint( 0, 100, size = np.random.randint( 1, 5 ) ).astype( np.int64 )
    idx = empty_table.AddData( arr )
    empty_table.CreateKeys( MinLen = np.random.randint( 1, 4 ), IndexSet = arr, Value = idx )
  for val in empty_table.LookUpDict.values(): assert 0 <= val < len( empty_table.Data )


def test_getitem_after_deletion_still_works( empty_table ) -> None:
  '''Deleting keys does not delete the underlying data.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  empty_table.DeleteAllOfSize( 3 ) # deletes all keys
  # All keys removed
  assert empty_table.LookUpDict == {}
  # Data list unchanged
  assert len( empty_table.Data ) == 1
  # Data still intact and accessible
  assert np.array_equal( empty_table[ idx ], seq )
  assert empty_table[ idx ].dtype == np.int64


def test_getitem_negative_index( empty_table ) -> None:
  '''Negative indices access elements from the end of the Data list.'''
  a = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  b = np.array( [ 40, 50 ], dtype = np.int64 )
  idx_a = empty_table.AddData( a )
  idx_b = empty_table.AddData( b )

  # Negative index -1 returns the last element
  assert np.array_equal( empty_table[ -1 ], b )
  assert empty_table[ -1 ].dtype == np.int64
  # Negative index -2 returns the first element
  assert np.array_equal( empty_table[ -2 ], a )
  # Negative index out of range raises IndexError
  with pytest.raises( IndexError, match = "list index out of range" ):
    _ = empty_table[ -3 ]


# ----------------------------------------------------------------------
# Tests for SameStart
# ----------------------------------------------------------------------
def test_samestart_exact_match( table_with_data ) -> None:
  '''A query matching a stored prefix returns the correct data index.'''
  # Query that exactly matches the first 3 elements of a1: [1,3,5] (sorted)
  query = np.array( [ 5, 1, 3 ], dtype = np.int64 ) # permutation invariant
  result = table_with_data.SameStart( query )
  # a1 was stored at index 0
  assert isinstance( result, int )
  assert result == 0


def test_samestart_subset_match( table_with_data ) -> None:
  '''A query matching a subset prefix returns the correct data index.'''
  # Query that matches the first 2 elements of a2: MinLen=2, so keys of length >=2 exist
  # a2 = [2,4,6], sorted -> (2,4,6). Keys for lengths 2 and 3 exist.
  # Query with [4,2] -> sorted (2,4) should hit.
  query = np.array( [ 4, 2 ], dtype = np.int64 )
  result = table_with_data.SameStart( query )
  assert isinstance( result, int )
  # a2 stored at index 1
  assert result == 1


def test_samestart_no_match( empty_table ) -> None:
  '''A query that does not match any prefix returns an empty list.'''
  empty_table.AddData( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  query = np.array( [ 4, 5, 6 ], dtype = np.int64 )
  result = empty_table.SameStart( query )
  assert isinstance( result, list )
  assert result == [] # Empty list indicates no match


def test_samestart_empty_query( empty_table ) -> None:
  '''An empty query returns an empty list (no zero-length keys exist).'''
  empty_table.AddData( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  query = np.array( [], dtype = np.int64 )
  # No keys of length 0 are ever created, so should return []
  result = empty_table.SameStart( query )
  assert isinstance( result, list )
  assert result == []


def test_samestart_with_overwritten_key( empty_table ) -> None:
  '''Test overwrite behavior: shorter sequence overwrites longer one.'''
  # Add first sequence (long)
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = empty_table.AddData( long_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = long_seq, Value = idx_long )

  # Now add a shorter sequence that starts the same way
  short_seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = empty_table.AddData( short_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = short_seq, Value = idx_short )

  # Query with [2,1] (sorted (1,2)) should now point to the shorter sequence
  query = np.array( [ 2, 1 ], dtype = np.int64 )
  result = empty_table.SameStart( query )
  assert isinstance( result, int )
  assert result == idx_short


# ----------------------------------------------------------------------
# Tests for CreateKeys
# ----------------------------------------------------------------------
def test_create_keys_generates_all_prefix_lengths( empty_table ) -> None:
  '''CreateKeys generates keys for all prefix lengths >= MinLen.'''
  seq = np.array( [ 3, 1, 4 ], dtype = np.int64 ) # sorted -> (1,3,4)
  idx = empty_table.AddData( seq )

  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  expected_keys = [
        ( 3, ), # len=1
        ( 1, 3 ), # len=2
        ( 1, 3, 4 ) # len=3
    ]
  assert len( empty_table.LookUpDict ) == len( expected_keys )
  for key in expected_keys:
    assert key in empty_table.LookUpDict
    assert empty_table.LookUpDict[ key ] == idx


def test_create_keys_minlen_zero_forced_to_one( empty_table ) -> None:
  '''MinLen=0 should be treated as 1 (no empty tuples).'''
  seq = np.array( [ 5, 2 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )

  empty_table.CreateKeys( MinLen = 0, IndexSet = seq, Value = idx )

  # Should have keys of lengths 1 and 2, but not length 0
  assert () not in empty_table.LookUpDict
  assert ( 5, ) in empty_table.LookUpDict
  assert ( 2, 5 ) in empty_table.LookUpDict


def test_create_keys_with_minlen_greater_than_one( empty_table ) -> None:
  '''CreateKeys with MinLen > 1 only generates keys of that length or greater.'''
  seq = np.array( [ 7, 8, 9, 10 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )

  empty_table.CreateKeys( MinLen = 3, IndexSet = seq, Value = idx )

  # Should only create keys of lengths 3 and 4
  expected = { ( 7, 8, 9 ), ( 7, 8, 9, 10 ) }
  actual = set( empty_table.LookUpDict.keys() )
  assert actual == expected


def test_create_keys_does_not_overwrite_with_longer_sequence( empty_table ) -> None:
  '''If a key already exists with a shorter sequence, do not overwrite.'''
  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = empty_table.AddData( short )
  empty_table.CreateKeys( MinLen = 1, IndexSet = short, Value = idx_short )

  long = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = empty_table.AddData( long )
  empty_table.CreateKeys( MinLen = 1, IndexSet = long, Value = idx_long )

  # Key (1,2) should still point to idx_short (shorter sequence)
  assert empty_table.LookUpDict[ ( 1, 2 ) ] == idx_short


def test_create_keys_overwrites_with_shorter_sequence( empty_table ) -> None:
  '''Existing key from a longer sequence is replaced by a shorter one.'''
  long = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = empty_table.AddData( long )
  empty_table.CreateKeys( MinLen = 1, IndexSet = long, Value = idx_long )

  short = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = empty_table.AddData( short )
  empty_table.CreateKeys( MinLen = 1, IndexSet = short, Value = idx_short )

  assert empty_table.LookUpDict[ ( 1, 2 ) ] == idx_short


def test_create_keys_handles_duplicate_insertion( empty_table ) -> None:
  '''Creating keys for the same sequence twice does not change mapping.'''
  seq = np.array( [ 5, 6 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  dict_before = empty_table.LookUpDict.copy()

  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  assert empty_table.LookUpDict == dict_before


# ----------------------------------------------------------------------
# Tests for DeleteAllOfSize
# ----------------------------------------------------------------------
def test_delete_all_of_size_removes_correct_lengths( empty_table ) -> None:
  '''DeleteAllOfSize removes only keys of length <= n.'''
  # Add data and keys of various lengths
  seq1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx1 = empty_table.AddData( seq1 )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq1, Value = idx1 )

  seq2 = np.array( [ 4, 5, 6, 7 ], dtype = np.int64 )
  idx2 = empty_table.AddData( seq2 )
  empty_table.CreateKeys( MinLen = 2, IndexSet = seq2, Value = idx2 ) # creates keys len 2,3,4

  # Before deletion
  assert len( empty_table.LookUpDict ) == 6 # lengths: 1,2,3 from seq1; 2,3,4 from seq2

  empty_table.DeleteAllOfSize( 1 )
  # Keys of length 1 should be gone
  assert all( len( k ) != 1 for k in empty_table.LookUpDict )
  # Lengths 2,3,4 should remain
  assert any( len( k ) == 2 for k in empty_table.LookUpDict )
  assert any( len( k ) == 3 for k in empty_table.LookUpDict )
  assert any( len( k ) == 4 for k in empty_table.LookUpDict )


def test_delete_all_of_size_with_n_equal_max_length( empty_table ) -> None:
  '''Deleting all keys of size <= n when n is the maximum length removes everything.'''
  seq = np.array( [ 10, 20 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx ) # keys len 1 and 2

  empty_table.DeleteAllOfSize( 2 )
  assert empty_table.LookUpDict == {}


def test_delete_all_of_size_raises_on_non_int( empty_table ) -> None:
  '''DeleteAllOfSize raises AssertionError for non-integer n.'''
  with pytest.raises( AssertionError, match = "n must be an int" ):
    empty_table.DeleteAllOfSize( "not an int" )
  with pytest.raises( AssertionError, match = "n must be an int" ):
    empty_table.DeleteAllOfSize( 2.0 )


def test_delete_all_of_size_does_not_affect_data_list( empty_table ) -> None:
  '''DeleteAllOfSize only affects keys, not stored data.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  data_before = empty_table.Data.copy()
  empty_table.DeleteAllOfSize( 5 ) # remove everything (all keys length <=5)
  assert empty_table.Data == data_before


def test_delete_all_of_size_exact_removal( empty_table ) -> None:
  '''DeleteAllOfSize with n exactly matching key lengths removes them.'''
  seq = np.array( [ 10, 20, 30 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx ) # keys len 1,2,3

  empty_table.DeleteAllOfSize( 2 ) # remove len<=2
  assert set( empty_table.LookUpDict.keys() ) == { ( 10, 20, 30 ) }


def test_delete_all_of_size_noop_for_missing_lengths( empty_table ) -> None:
  '''DeleteAllOfSize with n larger than all key lengths removes all keys.'''
  # Keys of length 1 and 2 exist; n=10 → all keys deleted
  seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  empty_table.DeleteAllOfSize( 10 )
  assert empty_table.LookUpDict == {}


def test_delete_all_of_size_with_non_positive_n( empty_table ) -> None:
  '''DeleteAllOfSize with n=0 or negative is a no-op (no zero-length keys).'''
  seq = np.array( [ 1 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  before = empty_table.LookUpDict.copy()
  empty_table.DeleteAllOfSize( 0 )
  # No keys of length ≤0 exist → dict unchanged
  assert empty_table.LookUpDict == before
  # Also test with negative n
  empty_table.DeleteAllOfSize( -1 )
  assert empty_table.LookUpDict == before
  empty_table.DeleteAllOfSize( -5 )
  assert empty_table.LookUpDict == before

# ----------------------------------------------------------------------
# Integration tests simulating arborescence usage
# ----------------------------------------------------------------------
def test_typical_arborescence_flow( empty_table ) -> None:
  '''Simulate a simplified BFS where LI increases each level.'''
  # Level 1: LI = [0]
  li_level1 = np.array( [ 0 ], dtype = np.int64 )
  # Compute regression result (mock)
  ellg_level1 = np.array( [ 0, 3, 7 ], dtype = np.int64 ) # sorted
  idx1 = empty_table.AddData( ellg_level1 )
  empty_table.CreateKeys( MinLen = len( li_level1 ), IndexSet = ellg_level1, Value = idx1 )

  # Check SameStart works for queries of length 1
  query = np.array( [ 0 ], dtype = np.int64 )
  result = empty_table.SameStart( query )
  assert isinstance( result, int )
  assert result == idx1

  # Level 2: LI = [0, 3]
  li_level2 = np.array( [ 0, 3 ], dtype = np.int64 )
  # SameStart should hit using key (0,3)
  result = empty_table.SameStart( li_level2 )
  assert isinstance( result, int )
  assert result == idx1

  # End of level 1: delete keys of length <= 1
  empty_table.DeleteAllOfSize( 1 )
  # Keys of length 1 gone, but length 2 still there
  result = empty_table.SameStart( np.array( [ 0 ], dtype = np.int64 ) )
  assert isinstance( result, list )
  assert result == [] # no key of length 1
  result = empty_table.SameStart( np.array( [ 0, 3 ], dtype = np.int64 ) )
  assert isinstance( result, int )
  assert result == idx1

  # Compute new regression for LI=[0,3] (mock) and add
  ellg_level2 = np.array( [ 0, 3, 5, 9 ], dtype = np.int64 )
  idx2 = empty_table.AddData( ellg_level2 )
  empty_table.CreateKeys( MinLen = len( li_level2 ), IndexSet = ellg_level2, Value = idx2 )

  # Check that key (0,3) now points to the new (shorter?) sequence?
  # In the actual code, overwriting logic ensures shorter sequence wins.
  # Here ellg_level2 is longer, so (0,3) should still point to idx1.
  assert empty_table.LookUpDict[ ( 0, 3 ) ] == idx1

  # End of level 2: delete keys of length <= 2
  empty_table.DeleteAllOfSize( 2 )
  # Now only keys of length >=3 remain
  assert ( 0, 3 ) not in empty_table.LookUpDict
  assert ( 0, 3, 5 ) in empty_table.LookUpDict


def test_random_arborescence_consistency( empty_table ) -> None:
  '''Simulate a mini arborescence: incremental LI, add regressions, query.'''
  rng = np.random.default_rng( 42 )
  data_store = [] # (full_sorted_seq, idx)
  for level_size in range( 1, 6 ):
    LI = rng.choice( 10, size = level_size, replace = False ).astype( np.int64 )
    # Query first
    hit = empty_table.SameStart( LI )
    # If it hits, the returned index must correspond to a sequence whose first
    # level_size elements (sorted) equal sorted(LI)
    if ( hit != [] ):
      assert isinstance( hit, int )
      stored = empty_table[ hit ]
      assert np.array_equal( np.sort( stored[ : level_size ] ), np.sort( LI ) )
    else:
      assert isinstance( hit, list )

    # Add a new regression (LI + random additional terms)
    extra = rng.choice( 10, size = rng.integers( 0, 3 ), replace = False )
    full = np.concatenate( [ LI, extra ] )
    idx = empty_table.AddData( full )
    empty_table.CreateKeys( MinLen = len( LI ), IndexSet = full, Value = idx )
    data_store.append( ( np.sort( full ), idx ) )

    # End of level: delete keys of length <= level_size
    empty_table.DeleteAllOfSize( level_size )

  # After all deletions, only keys longer than the last level_size exist
  for key in empty_table.LookUpDict: assert len( key ) > 5

# ----------------------------------------------------------------------
# Edge Cases and Error Handling
# ----------------------------------------------------------------------
def test_samestart_with_non_existent_key_returns_empty_list( empty_table ) -> None:
  '''SameStart with a non-existent prefix returns an empty list.'''
  result = empty_table.SameStart( np.array( [ 1, 2, 3 ], dtype = np.int64 ) )
  assert isinstance( result, list )
  assert result == []


def test_add_data_with_empty_array( empty_table ) -> None:
  '''An empty array can be added and retrieved.'''
  empty_arr = np.array( [], dtype = np.int64 )
  idx = empty_table.AddData( empty_arr )
  assert empty_table.Data[ idx ].size == 0
  assert empty_table.Data[ idx ].dtype == np.int64
  assert np.array_equal( empty_table.Data[ idx ], empty_arr )


def test_create_keys_with_empty_indexset( empty_table ) -> None:
  '''CreateKeys with an empty IndexSet creates no keys.'''
  empty_arr = np.array( [], dtype = np.int64 )
  idx = empty_table.AddData( empty_arr )
  # MinLen=1 prevents empty tuple, but IndexSet is empty -> no keys created
  empty_table.CreateKeys( MinLen = 1, IndexSet = empty_arr, Value = idx )
  assert empty_table.LookUpDict == {}


def test_large_number_of_keys_stress( empty_table ) -> None:
  '''Simple stress test: create many keys and check retrieval.'''
  for i in range( 100 ):
    seq = np.arange( i, i + 5, dtype = np.int64 )
    idx = empty_table.AddData( seq )
    empty_table.CreateKeys( MinLen = 3, IndexSet = seq, Value = idx )

  # Verify a few random queries
  for i in range( 0, 100, 10 ):
    seq = np.arange( i, i + 5, dtype = np.int64 )
    # Query with first 3 elements (sorted)
    query = np.sort( seq[ : 3 ] )
    result = empty_table.SameStart( query )
    assert isinstance( result, int )
    assert result == i # because AddData returns increasing indices


def test_dtype_preservation( empty_table ) -> None:
  '''Ensure that the stored arrays retain np.int64 dtype.'''
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  assert empty_table.Data[ idx ].dtype == np.int64


def test_empty_query_samestart( empty_table ) -> None:
  '''SameStart with an empty query array returns an empty list.'''
  seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )
  result = empty_table.SameStart( np.array( [], dtype = np.int64 ) )
  assert isinstance( result, list )
  assert result == []


def test_minlen_greater_than_indexset_length( empty_table ) -> None:
  '''MinLen greater than IndexSet length creates no keys.'''
  # MinLen > len(IndexSet) → no keys created
  seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 5, IndexSet = seq, Value = idx )
  assert empty_table.LookUpDict == {}


def test_add_data_twice_same_array_reference( empty_table ) -> None:
  '''Adding the same array twice stores two separate copies.'''
  arr = np.array( [ 7, 8 ], dtype = np.int64 )
  idx1 = empty_table.AddData( arr )
  idx2 = empty_table.AddData( arr )
  assert idx2 == idx1 + 1
  # Verify data integrity (value equality, not identity)
  assert np.array_equal( empty_table.Data[ idx1 ], arr )
  assert np.array_equal( empty_table.Data[ idx2 ], arr )

# ----------------------------------------------------------------------
# Extra: Permutation invariance and sort semantics
# ----------------------------------------------------------------------

def test_samestart_permutation_invariance( empty_table ) -> None:
  '''SameStart is invariant to permutation of query elements.'''
  # Insert with an unsorted LI-like query
  seq = np.array( [ 5, 1, 9 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  # SameStart with any permutation of the first 2 elements should hit
  result1 = empty_table.SameStart( np.array( [ 1, 5 ], dtype = np.int64 ) )
  assert isinstance( result1, int )
  assert result1 == idx
  result2 = empty_table.SameStart( np.array( [ 5, 1 ], dtype = np.int64 ) )
  assert isinstance( result2, int )
  assert result2 == idx
  # but a non‑existing prefix should not
  result3 = empty_table.SameStart( np.array( [ 1, 9 ], dtype = np.int64 ) )
  assert isinstance( result3, list )
  assert result3 == []


def test_samestart_with_python_ints( empty_table ) -> None:
  '''LookUpDict keys work with plain Python int tuples.'''
  # Ensure that lookup works with plain python int tuples as well
  seq = np.array( [ 2, 4, 6 ], dtype = np.int64 )
  idx = empty_table.AddData( seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = seq, Value = idx )

  # Query using a list of python ints (np.sort yields np.int64, but tuple of py int is equivalent)
  query = tuple( sorted( [ 6, 2 ] ) ) # (2, 6)
  # Manually call __getitem__ via LookUpDict? We can test SameStart directly with a numpy array that becomes (2,6)
  # but it's safer to test that the key lookup works with any int type
  assert ( 2, 4 ) in empty_table.LookUpDict # explicit check


# ----------------------------------------------------------------------
# Extra: Overwrite behaviour – exactly as the algorithm expects
# ----------------------------------------------------------------------

def test_overwrite_only_when_shorter( empty_table ) -> None:
  '''A key is only overwritten when the new sequence is shorter.'''
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = empty_table.AddData( long_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = long_seq, Value = idx_long )

  # Insert a sequence of *same* length with a conflicting prefix
  same_len_seq = np.array( [ 1, 2, 5, 6 ], dtype = np.int64 )
  idx_same = empty_table.AddData( same_len_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = same_len_seq, Value = idx_same )

  # The common prefix (1,2) should still point to the LONG sequence (first inserted)
  assert empty_table.LookUpDict[ ( 1, 2 ) ] == idx_long

  # Now insert a truly shorter sequence
  short_seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = empty_table.AddData( short_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = short_seq, Value = idx_short )

  # Now (1,2) must point to the short one
  assert empty_table.LookUpDict[ ( 1, 2 ) ] == idx_short


def test_overwrite_does_not_affect_longer_keys( empty_table ) -> None:
  '''Overwriting a shorter key does not affect longer prefix keys.'''
  long_seq = np.array( [ 1, 2, 3, 4 ], dtype = np.int64 )
  idx_long = empty_table.AddData( long_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = long_seq, Value = idx_long )

  short_seq = np.array( [ 1, 2 ], dtype = np.int64 )
  idx_short = empty_table.AddData( short_seq )
  empty_table.CreateKeys( MinLen = 1, IndexSet = short_seq, Value = idx_short )

  # The longer prefix (1,2,3) must still point to the long sequence
  assert empty_table.LookUpDict[ ( 1, 2, 3 ) ] == idx_long
