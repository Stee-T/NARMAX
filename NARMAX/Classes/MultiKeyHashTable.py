import numpy as np

from typing import Union, Sequence
from numpy.typing import NDArray

class MultiKeyHashTable:
  """Class used for LG which contains a list storing the data and a hash table containing the keys to list-index mapping"""
  def __init__( self ) -> None:
    self.Data: list[ NDArray[ np.int64 ] ] = [] # List for data storage
    self.LookUpDict: dict[ tuple[ int ], int ] = {} # HashTable for quick lookup

  # --------------------------------------------------- [] operator ----------------------------------------------------
  def __getitem__( self, index: int ) -> NDArray[ np.int64 ]:
    '''Get an item from the Data list by its list index: self.MultiKeyHashTable[ index ] '''
    return ( self.Data[ index ] )

  # ---------------------------------------------------- Same Start ----------------------------------------------------
  # TODO: this should return an Optional[ int ] to be clearer rather than an empty list
  def SameStart( self, Item: NDArray[ np.int64 ] ) -> Union[ list[ int ], int ]:
    ''' Getter retrieving an item from the Data list indirectly via the LookUpDict by checking if Item matches the start of any element.
    This is the check performed before node creations and during the rFOrLSR.
    
    ### Input:
    - `Item`: 1D int array

    ### Output:
    - `Out`: If a corresponding LG element is found, return it (int) else an empty list ([])
    '''
    key = tuple( np.sort( Item ) )
    if ( key in self.LookUpDict ): return ( self.LookUpDict[ key ] ) # return index in Data
    else: return ( [] ) # no matching term was found during the iteration


  # ----------------------------------------------------- Add Data -----------------------------------------------------
  def AddData( self, Item: NDArray[ np.int64 ] ) -> int:
    '''Add an item to the LG
    
    ### Input:
    - `Item`: 1D int array to be added to the Data list

    ### Output:
    - `Out`: (int) index where the item was added in the data list
    '''
    self.Data.append( Item )
    return ( len( self.Data ) - 1 ) # ( -1 since zero-based )
  

  # ---------------------------------------------------- Create Keys ---------------------------------------------------
  def CreateKeys( self, MinLen: int, IndexSet: Union[ Sequence[ int ], NDArray[ np.int64 ] ], Value: int ) -> None:
    '''Create all keys in the LookUpDict pointing to the passed Value
    ### Inputs:
    - `MinLen`: (int) containing the minimum key length
    - `IndexSet`: (iterable of int) containing the index set to be used as keys
    - `Value`: (int) containing the value to be stored in the LookUpDict, being the index in the Data list
    '''
    if ( MinLen == 0 ): MinLen = 1 # Root only, prevents [:0] from creating empty tuple. no terms are lost due to the +1 in the loop
    
    for i in range( MinLen, len( IndexSet ) + 1 ): # Create entries of all allowed lengths, +1 to compensate the [:i] eliminating one element
      key = tuple( np.sort( IndexSet[:i] ) ) # Make the key permutaion invariant
      if ( ( key not in self.LookUpDict ) or ( len( self.Data[ self.LookUpDict[ key ] ] ) > len( IndexSet ) ) ): # if unknown or inserted is shorter: overwrite
        self.LookUpDict[ key ] = Value # append the sorted list to lookup dictionary with self.LG's last index
    
    
  # ------------------------------------------------ Delete All Of Size ------------------------------------------------
  def DeleteAllOfSize( self, n: int ) -> None:
    '''Delete all Look-up table items with size n. Used at the end of every arbo level to gain a bit of memory.
    Since len( LI ) == n+1 at the next level, there won't ever be a query of size < n+1, so delete those.
    ### Inputs:
    - `n`: int
    '''
    if ( not isinstance( n, int ) ): raise AssertionError( "n must be an int" )

    for reg in list( self.LookUpDict.keys() ): # Act on the keys, list since deleting while iterating is illegal
      if ( len( reg ) <= n ): del self.LookUpDict[ reg ] # remove the entry
