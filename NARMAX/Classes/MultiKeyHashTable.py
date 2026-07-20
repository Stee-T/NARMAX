import sys
import math
import bisect
import numpy as np
from typing import Iterator
from numpy.typing import NDArray


class LevelTable:
  __slots__ = ( 'Fingerprints', 'Indices', 'Capacity', 'Mask', 'Size' )

  def __init__( self, Capacity: int ) -> None:
    self.Capacity: int = Capacity
    self.Mask: int = Capacity - 1
    self.Size: int = 0
    self.Fingerprints: NDArray[ np.uint64 ] = np.empty( Capacity, dtype = np.uint64 )
    self.Indices: NDArray[ np.int64 ] = np.full( Capacity, -1, dtype = np.int64 )


class MultiKeyHashTable:
  def __init__( self, IntType: type = np.uint32 ) -> None:
    self.IntType: np.dtype = np.dtype( IntType )
    self.InitialTableCap: int = 16
    self.Tables: list[ LevelTable | None ] = []

    self.DataCap: int = 1024
    self.FlatData: NDArray = np.empty( self.DataCap, dtype = self.IntType )
    self.Offsets: list[ int ] = [ 0 ]
    self.DataCount: int = 0

  @staticmethod
  def ComputeRank( SortedList: list[ int ] ) -> int:
    Rank = 0
    for k, e in enumerate( SortedList, 1 ):
      assert e >= 0, "negative elements are not supported by combinatorial rank"
      if ( e >= k ): Rank += math.comb( e, k )
    return Rank & 0xFFFFFFFFFFFFFFFF

  # ------------------------------------------------------ length ------------------------------------------------------
  def __len__( self ) -> int: return self.DataCount

  # ------------------------------------------------ Memory Usage ---------------------------------------------------
  def get_memory_usage( self ) -> dict[ str, int ]:
    MemArray: int = 0
    MemTables: int = 0
    MemOther: int = 0

    MemOther += sys.getsizeof( self )

    MemArray += sys.getsizeof( self.FlatData )
    MemArray += sys.getsizeof( self.Offsets )
    for o in self.Offsets:
      if ( not ( isinstance( o, int ) and - 5 <= o <= 256 ) ): MemArray += sys.getsizeof( o )

    MemTables += sys.getsizeof( self.Tables )
    for tbl in self.Tables:
      if ( tbl is None ): continue
      MemTables += sys.getsizeof( tbl )
      MemTables += sys.getsizeof( tbl.Fingerprints )
      MemTables += sys.getsizeof( tbl.Indices )
      MemTables += sys.getsizeof( tbl.Capacity )
      MemTables += sys.getsizeof( tbl.Mask )
      MemTables += sys.getsizeof( tbl.Size )

    MemOther += sys.getsizeof( self.IntType )
    MemOther += sys.getsizeof( self.DataCap )
    MemOther += sys.getsizeof( self.DataCount )
    MemOther += sys.getsizeof( self.InitialTableCap )

    return {
        'array': MemArray,
        'tables': MemTables,
        'other': MemOther,
        'total': MemArray + MemTables + MemOther,
    }

  # --------------------------------------------------- iteration ------------------------------------------------------
  def __iter__( self ) -> Iterator[ NDArray ]:
    for i in range( self.DataCount ): yield self[ i ]

  # --------------------------------------------------- [] operator ----------------------------------------------------
  def __getitem__( self, Index: int ) -> NDArray:
    if ( Index < 0 ): Index += self.DataCount
    if ( ( Index < 0 ) or ( Index >= self.DataCount ) ): raise IndexError( "list index out of range" )
    Start: int = self.Offsets[ Index ]
    Length: int = self.Offsets[ Index + 1 ] - self.Offsets[ Index ]
    return self.FlatData[ Start : Start + Length ].copy()

  # ---------------------------------------------------- Same Start ----------------------------------------------------
  def SameStart( self, Item: NDArray[ np.integer ] ) -> int | None:
    ''' Getter retrieving an item from the Data list indirectly via the LookUpDict by checking if Item matches the start of any element.
    This is the check performed before node creations and during the rFOrLSR.

    ### Input:
    - `Item`: 1D int array

    ### Output:
    - `Out`: If a corresponding LG element is found, return it (int) else an empty list ([])
    '''
    if ( ( len( Item ) >= len( self.Tables ) ) or ( self.Tables[ len( Item ) ] is None ) ): return None # R[1/3]
    Lst = Item.tolist()
    Lst.sort()
    Rank = self.ComputeRank( Lst )
    Tbl = self.Tables[ len( Item ) ]
    Slot = int( Rank & Tbl.Mask )
    while ( Tbl.Indices[ Slot ] != -1 ): # Slot occupied, -1 is sentinel
      if ( Tbl.Fingerprints[ Slot ] == Rank ): return int( Tbl.Indices[ Slot ] ) # R[2/3]
      Slot = ( Slot + 1 ) & Tbl.Mask
    return None # R[3/3]

  # ----------------------------------------------------- Add Data -----------------------------------------------------
  def AddData( self, Item: NDArray[ np.integer ] ) -> int:
    '''Add an item to the LG

    ### Input:
    - `Item`: 1D int array to be added to the Data list

    ### Output:
    - `Out`: (int) index where the item was added in the data list
    '''
    lenItem = len( Item )
    if ( self.Offsets[ -1 ] + lenItem > self.DataCap ): # Doesn't fit into current array
      self.DataCap = max( self.DataCap * 2, self.Offsets[ -1 ] + lenItem )
      NewBuf = np.empty( self.DataCap, dtype = self.IntType )
      NewBuf[ : self.Offsets[ -1 ] ] = self.FlatData[ : self.Offsets[ -1 ] ]
      self.FlatData = NewBuf

    self.FlatData[ self.Offsets[ -1 ] : self.Offsets[ -1 ] + lenItem ] = Item
    self.DataCount += 1
    self.Offsets.append( self.Offsets[ -1 ] + lenItem )
    return self.DataCount -1 # return index, -1 since zero-based

  # ---------------------------------------------------- Create Keys ---------------------------------------------------
  def CreateKeys( self, MinLen: int, IndexSet: NDArray[ np.integer ], Value: int ) -> None:
    '''Create all keys in the LookUpDict pointing to the passed Value
    ### Inputs:
    - `MinLen`: (int) containing the minimum key length
    - `IndexSet`: (iterable of int) containing the index set to be used as keys
    - `Value`: (int) containing the value to be stored in the LookUpDict, being the index in the Data list
    '''
    if ( MinLen == 0 ): MinLen = 1
    MaxLen: int = len( IndexSet )
    if ( len( self.Tables ) <= MaxLen ): self.Tables.extend( [ None ] * ( MaxLen - len( self.Tables ) + 1 ) )

    Lst = IndexSet.tolist()
    Prefix = Lst[ : MinLen ]

    for i in range( MinLen, MaxLen + 1 ):
      if ( i > MinLen ): bisect.insort( Prefix, Lst[ i - 1 ] )
      Rank = self.ComputeRank( Prefix )
      if ( self.Tables[ i ] is None ): self.Tables[ i ] = LevelTable( self.InitialTableCap )
      self.Insert( i, Rank, Value )

  # --------------------------------------------------- insert ------------------------------------------------------
  def Insert( self, Level: int, Rank: int, Idx: int ) -> None:
    Tbl: LevelTable = self.Tables[ Level ]
    Slot: int = int( Rank & Tbl.Mask )
    while ( Tbl.Indices[ Slot ] != -1 ):
      if ( Tbl.Fingerprints[ Slot ] == Rank ):
        StoredLen = self.Offsets[ Tbl.Indices[ Slot ] + 1 ] - self.Offsets[ Tbl.Indices[ Slot ] ]
        NewLen = self.Offsets[ Idx + 1 ] - self.Offsets[ Idx ]
        if ( StoredLen > NewLen ): Tbl.Indices[ Slot ] = Idx
        return
      Slot = ( Slot + 1 ) & Tbl.Mask
    Tbl.Fingerprints[ Slot ] = Rank
    Tbl.Indices[ Slot ] = Idx
    Tbl.Size += 1
    if ( Tbl.Size >= int( Tbl.Capacity * 0.75 ) ): self.Resize( Level )

  # --------------------------------------------------- resize ------------------------------------------------------
  def Resize( self, Level: int ) -> None:
    OldTbl: LevelTable = self.Tables[ Level ]
    NewCap: int = OldTbl.Capacity * 2
    NewTbl: LevelTable = LevelTable( NewCap )
    OccMask: NDArray[ np.bool_ ] = OldTbl.Indices != -1
    OldRanks: NDArray[ np.uint64 ] = OldTbl.Fingerprints[ OccMask ]
    OldIdxs: NDArray[ np.int64 ] = OldTbl.Indices[ OccMask ]

    for Rank, Idx in zip( OldRanks, OldIdxs ):
      Slot: int = int( Rank & NewTbl.Mask )
      while ( NewTbl.Indices[ Slot ] != -1 ):
        if ( NewTbl.Fingerprints[ Slot ] == Rank ):
          StoredLen = self.Offsets[ NewTbl.Indices[ Slot ] + 1 ] - self.Offsets[ NewTbl.Indices[ Slot ] ]
          NewLen = self.Offsets[ Idx + 1 ] - self.Offsets[ Idx ]
          if ( StoredLen > NewLen ): NewTbl.Indices[ Slot ] = Idx
          break
        Slot = ( Slot + 1 ) & NewTbl.Mask
      else:
        NewTbl.Fingerprints[ Slot ] = Rank
        NewTbl.Indices[ Slot ] = Idx
    NewTbl.Size = OldTbl.Size
    self.Tables[ Level ] = NewTbl

  # ------------------------------------------------ Delete All Of Size ------------------------------------------------
  def DeleteAllOfSize( self, n: int ) -> None:
    '''Delete all Look-up tables of size n. Used at the end of every arbo level to gain a bit of memory.
    Since len( LI ) == n+1 at the next level, there won't ever be a query of size < n+1, so delete those.
    ### Inputs:
    - `n`: int
    '''
    if ( not isinstance( n, int ) ): raise TypeError( "n must be an int" )
    for i in range( 0, n + 1 ):
      if ( i < len( self.Tables ) ): self.Tables[ i ] = None
