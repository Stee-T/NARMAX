import sys
import math
import bisect
import numpy as np
from numpy.typing import NDArray

class _LevelTable:
  __slots__ = ( 'fingerprints', 'indices', 'occupied', 'capacity', 'mask', 'size' )

  def __init__( self, capacity: int ) -> None:
    self.capacity = capacity
    self.mask = capacity - 1
    self.size = 0
    self.fingerprints = np.empty( capacity, dtype = np.uint64 )
    self.indices = np.empty( capacity, dtype = np.int64 )
    self.occupied = np.zeros( capacity, dtype = np.int8 )


class MultiKeyHashTable:
  def __init__( self, int_type: type = np.uint32 ) -> None:
    self.int_type: np.dtype = np.dtype( int_type )
    self._initial_table_cap: int = 16
    self.tables: list[ _LevelTable | None ] = []

    self._data_cap: int = 1024
    self._flat_data: NDArray = np.empty( self._data_cap, dtype = self.int_type )
    self._data_len: int = 0
    self._offsets: list[ int ] = [ 0 ]
    self._lengths: list[ int ] = []
    self._data_count: int = 0

  @staticmethod
  def _compute_rank( sorted_lst: list[ int ] ) -> int:
    rank = 0
    for k, e in enumerate( sorted_lst, 1 ):
      if ( e < 0 ): return hash( tuple( sorted_lst ) ) & 0xFFFFFFFFFFFFFFFF # R[1/2]
      if ( e >= k ): rank += math.comb( e, k )
    return rank & 0xFFFFFFFFFFFFFFFF # R[2/2]

  # ------------------------------------------------------ length ------------------------------------------------------
  def __len__( self ) -> int: return self._data_count

  # ------------------------------------------------ Memory Usage ---------------------------------------------------
  def get_memory_usage( self ) -> dict:
    mem_array = 0
    mem_tables = 0
    mem_other = 0

    mem_other += sys.getsizeof( self )

    mem_array += sys.getsizeof( self._flat_data )
    mem_array += sys.getsizeof( self._offsets )
    for o in self._offsets:
      if ( not ( isinstance( o, int ) and - 5 <= o <= 256 ) ): mem_array += sys.getsizeof( o )
    mem_array += sys.getsizeof( self._lengths )
    for l in self._lengths:
      if ( not ( isinstance( l, int ) and - 5 <= l <= 256 ) ): mem_array += sys.getsizeof( l )

    mem_tables += sys.getsizeof( self.tables )
    for tbl in self.tables:
      if ( tbl is None ): continue
      mem_tables += sys.getsizeof( tbl )
      mem_tables += sys.getsizeof( tbl.fingerprints )
      mem_tables += sys.getsizeof( tbl.indices )
      mem_tables += sys.getsizeof( tbl.occupied )
      mem_tables += sys.getsizeof( tbl.capacity )
      mem_tables += sys.getsizeof( tbl.mask )
      mem_tables += sys.getsizeof( tbl.size )

    mem_other += sys.getsizeof( self.int_type )
    mem_other += sys.getsizeof( self._data_cap )
    mem_other += sys.getsizeof( self._data_count )
    mem_other += sys.getsizeof( self._data_len )
    mem_other += sys.getsizeof( self._initial_table_cap )

    return {
        'array': mem_array,
        'tables': mem_tables,
        'other': mem_other,
        'total': mem_array + mem_tables + mem_other,
    }

  # --------------------------------------------------- iteration ------------------------------------------------------
  def __iter__( self ):
    for i in range( self._data_count ): yield self[ i ]

  # --------------------------------------------------- [] operator ----------------------------------------------------
  def __getitem__( self, index: int ) -> NDArray:
    if ( ( index < 0 ) or ( index >= self._data_count ) ): raise IndexError( "list index out of range" )
    start = self._offsets[ index ]
    length = self._lengths[ index ]
    return self._flat_data[ start : start + length ].copy()

  # ---------------------------------------------------- Same Start ----------------------------------------------------
  def SameStart( self, Item: NDArray[ np.integer ] ) -> int | None:
    ''' Getter retrieving an item from the Data list indirectly via the LookUpDict by checking if Item matches the start of any element.
    This is the check performed before node creations and during the rFOrLSR.

    ### Input:
    - `Item`: 1D int array

    ### Output:
    - `Out`: If a corresponding LG element is found, return it (int) else an empty list ([])
    '''
    if ( ( len( Item ) >= len( self.tables ) ) or ( self.tables[ len( Item ) ] is None ) ): return None # R[1/3]
    lst = Item.tolist()
    lst.sort()
    rank = self._compute_rank( lst )
    tbl = self.tables[ len( Item ) ]
    slot = int( rank & tbl.mask )
    while ( tbl.occupied[ slot ] ):
      if ( tbl.fingerprints[ slot ] == rank ): return int( tbl.indices[ slot ] ) # R[2/3]
      slot = ( slot + 1 ) & tbl.mask
    return None # R[3/3]

  # ----------------------------------------------------- Add Data -----------------------------------------------------
  def AddData( self, Item: NDArray[ np.integer ] ) -> int:
    '''Add an item to the LG

    ### Input:
    - `Item`: 1D int array to be added to the Data list

    ### Output:
    - `Out`: (int) index where the item was added in the data list
    '''
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
    '''Create all keys in the LookUpDict pointing to the passed Value
    ### Inputs:
    - `MinLen`: (int) containing the minimum key length
    - `IndexSet`: (iterable of int) containing the index set to be used as keys
    - `Value`: (int) containing the value to be stored in the LookUpDict, being the index in the Data list
    '''
    if ( MinLen == 0 ): MinLen = 1
    max_len = len( IndexSet )
    if ( len( self.tables ) <= max_len ): self.tables.extend( [ None ] * ( max_len - len( self.tables ) + 1 ) )

    lst = IndexSet.tolist()
    prefix = lst[ : MinLen ]

    for i in range( MinLen, max_len + 1 ):
      if ( i > MinLen ): bisect.insort( prefix, lst[ i - 1 ] )
      rank = self._compute_rank( prefix )
      if ( self.tables[ i ] is None ): self.tables[ i ] = _LevelTable( self._initial_table_cap )
      self._insert( i, rank, Value )

  # --------------------------------------------------- insert ------------------------------------------------------
  def _insert( self, level: int, rank: int, idx: int ) -> None:
    tbl = self.tables[ level ]
    slot = int( rank & tbl.mask )
    while ( tbl.occupied[ slot ] ):
      if ( tbl.fingerprints[ slot ] == rank ):
        stored_len = self._lengths[ tbl.indices[ slot ] ]
        new_len = self._lengths[ idx ]
        if ( stored_len > new_len ): tbl.indices[ slot ] = idx
        return
      slot = ( slot + 1 ) & tbl.mask
    tbl.fingerprints[ slot ] = rank
    tbl.indices[ slot ] = idx
    tbl.occupied[ slot ] = 1
    tbl.size += 1
    if ( tbl.size >= int( tbl.capacity * 0.75 ) ): self._resize( level )

  # --------------------------------------------------- resize ------------------------------------------------------
  def _resize( self, level: int ) -> None:
    old_tbl = self.tables[ level ]
    new_cap = old_tbl.capacity * 2
    new_tbl = _LevelTable( new_cap )
    occ_mask = old_tbl.occupied.astype( bool )
    old_ranks = old_tbl.fingerprints[ occ_mask ]
    old_idxs = old_tbl.indices[ occ_mask ]

    for rank, idx in zip( old_ranks, old_idxs ):
      slot = int( rank & new_tbl.mask )
      while ( new_tbl.occupied[ slot ] ):
        if ( new_tbl.fingerprints[ slot ] == rank ):
          stored_len = self._lengths[ new_tbl.indices[ slot ] ]
          new_len = self._lengths[ idx ]
          if ( stored_len > new_len ): new_tbl.indices[ slot ] = idx
          break
        slot = ( slot + 1 ) & new_tbl.mask
      else:
        new_tbl.fingerprints[ slot ] = rank
        new_tbl.indices[ slot ] = idx
        new_tbl.occupied[ slot ] = 1
    new_tbl.size = old_tbl.size
    self.tables[ level ] = new_tbl

  # ------------------------------------------------ Delete All Of Size ------------------------------------------------
  def DeleteAllOfSize( self, n: int ) -> None:
    '''Delete all Look-up tables of size n. Used at the end of every arbo level to gain a bit of memory.
    Since len( LI ) == n+1 at the next level, there won't ever be a query of size < n+1, so delete those.
    ### Inputs:
    - `n`: int
    '''
    if ( not isinstance( n, int ) ): raise TypeError( "n must be an int" )
    for i in range( 0, n + 1 ):
      if ( i < len( self.tables ) ): self.tables[ i ] = None
