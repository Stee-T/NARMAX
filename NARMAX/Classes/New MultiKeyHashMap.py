import numpy as np
from numpy.typing import NDArray

class _LevelTable:
  '''Internal hash table for a single sequence length level.'''
  __slots__ = ( 'fingerprints', 'indices', 'occupied', 'capacity', 'size', 'mask' )
  def __init__( self, capacity: int ) -> None:
    '''Initialize a level table with the given capacity.

    Args:
        capacity: The capacity of the table (must be a power of two).
    '''
    self.capacity: int = capacity
    self.mask: int = capacity - 1
    self.fingerprints: NDArray[ np.uint64 ] = np.zeros( capacity, dtype = np.uint64 )
    self.indices: NDArray[ np.int32 ] = np.full( capacity, -1, dtype = np.int32 )
    self.occupied: NDArray[ np.uint8 ] = np.zeros( capacity, dtype = np.uint8 )
    self.size: int = 0

class MultiKeyHashTable:
  '''Length-partitioned open-addressing hash table with contiguous sequence storage.
  Replaces dict/tuple overhead with flat NDArrays. O(1) level deletion.
  '''
  def __init__( self, int_type: type = np.uint32, initial_table_cap: int = 1 << 16 ) -> None:
    '''Initialize the hash table.

    Args:
        int_type: The numpy integer type for storing data.
        initial_table_cap: Initial capacity for each level table (must be power of two).
    '''
    self.int_type: np.dtype = np.dtype( int_type )
    self.tables: list[ _LevelTable | None ] = []
    self._initial_table_cap: int = initial_table_cap

    # Contiguous data storage
    self._data_cap: int = 1 << 20
    self._flat_data: NDArray = np.empty( self._data_cap, dtype = self.int_type )
    self._data_len: int = 0
    self._offsets: list[ int ] = [ 0 ]
    self._lengths: list[ int ] = []
    self._data_count: int = 0

  # --------------------------------------------------- [] operator ----------------------------------------------------
  def __getitem__( self, index: int ) -> NDArray:
    '''Get an item from the data storage by its index: self.table[ index ].'''
    start = self._offsets[ index ]
    length = self._lengths[ index ]
    return self._flat_data[ start : start + length ].copy()

  # ---------------------------------------------------- Same Start ----------------------------------------------------
  def SameStart( self, Item: NDArray[ np.integer ] ) -> int | None:
    '''Lookup unordered sequence. Returns Data index or None.'''
    length = len( Item )
    if ( ( length >= len( self.tables ) ) or ( self.tables[ length ] is None ) ): return None # R[1/3]

    sorted_item = np.sort( Item )
    h = self._commutative_hash( sorted_item )
    tbl = self.tables[ length ]
    slot = int( h & tbl.mask )

    while ( tbl.occupied[ slot ] ):
      if ( tbl.fingerprints[ slot ] == h ):
        if ( self._verify_match( tbl.indices[ slot ], sorted_item ) ): return int( tbl.indices[ slot ] ) # R[2/3]
      slot = ( slot + 1 ) & tbl.mask
    return None # R[3/3]

  # ----------------------------------------------------- Add Data -----------------------------------------------------
  def AddData( self, Item: NDArray[ np.integer ] ) -> int:
    '''Append sequence to contiguous storage. Returns index.'''
    n = len( Item )
    if ( self._data_len + n > self._data_cap ):
      self._data_cap = max( self._data_cap * 2, self._data_len + n )
      new_buf = np.empty( self._data_cap, dtype = self.int_type )
      new_buf[ : self._data_len ] = self._flat_data[ : self._data_len ]
      self._flat_data = new_buf

    self._flat_data[ self._data_len : self._data_len + n ] = Item
    self._data_len += n
    idx = self._data_count
    self._data_count += 1
    self._offsets.append( self._data_len )
    self._lengths.append( n )
    return idx

  # ---------------------------------------------------- Create Keys ---------------------------------------------------
  def CreateKeys( self, MinLen: int, IndexSet: NDArray[ np.integer ], Value: int ) -> None:
    '''Insert all prefix lengths [MinLen, len(IndexSet)] into partitioned tables.'''
    if ( MinLen == 0 ): MinLen = 1
    max_len = len( IndexSet )
    if ( len( self.tables ) <= max_len ): self.tables.extend( [ None ] * ( max_len - len( self.tables ) + 1 ) )

    for i in range( MinLen, max_len + 1 ):
      prefix = np.sort( IndexSet[ : i ] )
      h = self._commutative_hash( prefix )
      if ( self.tables[ i ] is None ): self.tables[ i ] = _LevelTable( self._initial_table_cap )
      self._insert( i, h, Value, prefix )

  # ------------------------------------------------ Delete All Of Size ------------------------------------------------
  def DeleteAllOfSize( self, n: int ) -> None:
    '''O(1) level cleanup. Drops all keys of length n.'''
    if ( not isinstance( n, int ) ): raise TypeError( "n must be an int" )
    if ( n < len( self.tables ) ): self.tables[ n ] = None

  # --------------------------------------------------- Internals ------------------------------------------------------
  @staticmethod
  def _commutative_hash( arr: NDArray[ np.integer ] ) -> np.uint64:
    '''Compute a commutative hash of an integer array (order-independent).'''
    u = arr.astype( np.uint64 )
    mixed = ( u * np.uint64( 0x9e3779b97f4a7c15 ) ) ^ ( u >> np.uint64( 33 ) )
    return np.bitwise_xor.reduce( mixed, initial = np.uint64( 0 ) )

  def _verify_match( self, idx: int, query: NDArray[ np.integer ] ) -> bool:
    '''Verify that the stored sequence at idx matches the query.'''
    start = self._offsets[ idx ]
    length = self._lengths[ idx ]
    if ( length != len( query ) ): return False # R[1/2]
    return np.array_equal( self._flat_data[ start : start + length ], query ) # R[2/2]

  def _insert( self, level: int, h: np.uint64, idx: int, seq: NDArray[ np.integer ] ) -> None:
    '''Insert a key-value pair into the level table.'''
    tbl = self.tables[ level ]
    slot = int( h & tbl.mask )
    while ( tbl.occupied[ slot ] ):
      if ( tbl.fingerprints[ slot ] == h and self._verify_match( tbl.indices[ slot ], seq ) ): return # Duplicate; BFS guarantees first insertion is optimal
      slot = ( slot + 1 ) & tbl.mask

    tbl.fingerprints[ slot ] = h
    tbl.indices[ slot ] = idx
    tbl.occupied[ slot ] = 1
    tbl.size += 1

    if ( tbl.size >= int( tbl.capacity * 0.75 ) ): self._resize( level )

  def _resize( self, level: int ) -> None:
    '''Resize the level table at the given level to double its capacity.'''
    old_tbl = self.tables[ level ]
    new_cap = old_tbl.capacity * 2
    new_tbl = _LevelTable( new_cap )

    occ_mask = old_tbl.occupied.astype( bool )
    old_fps = old_tbl.fingerprints[ occ_mask ]
    old_idxs = old_tbl.indices[ occ_mask ]

    for fp, idx in zip( old_fps, old_idxs ):
      slot = int( fp & new_tbl.mask )
      while ( new_tbl.occupied[ slot ] ): slot = ( slot + 1 ) & new_tbl.mask
      new_tbl.fingerprints[ slot ] = fp
      new_tbl.indices[ slot ] = idx
      new_tbl.occupied[ slot ] = 1
    new_tbl.size = old_tbl.size
    self.tables[ level ] = new_tbl
