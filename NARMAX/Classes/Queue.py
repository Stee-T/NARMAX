from collections import deque
import numpy as np
from numpy.typing import NDArray

class QueueEmptyError( Exception ):
  '''Raised when trying to get from an empty queue.'''
  pass


class Queue:
  '''Simple, unbounded FIFO queue wrapping collections.deque.
  Heavily modified from the original Python 3.10 implementation.
  Eliminated unwanted operations and added peek().
  https://github.com/python/cpython/blob/3.10/Lib/queue.py
  '''

  def __init__( self ) -> None:
    '''Initialize an empty queue.'''
    self.data: deque[ NDArray[ np.int64 ] ] = deque()

  def put( self, item: NDArray[ np.int64 ] ) -> None:
    '''Put the item on the queue.'''
    if ( ( not isinstance( item, np.ndarray ) ) or ( not np.issubdtype( item.dtype, np.integer ) ) ):
      raise TypeError( "Only integer numpy arrays are allowed" )
    self.data.append( item )

  def get( self ) -> NDArray[ np.int64 ]:
    '''Remove and return an item from the queue.

    Raises:
        QueueEmptyError: If the queue is empty.
    '''
    if ( not self.data ): raise QueueEmptyError( "The Queue is empty" )
    return self.data.popleft()

  def is_empty( self ) -> bool:
    '''Return True if the queue is empty, False otherwise.'''
    return len( self.data ) == 0

  def clear( self ) -> None:
    '''Clear the queue.'''
    self.data.clear()

  def size( self ) -> int:
    '''Return the number of items in the queue.'''
    return len( self.data )

  def peek( self ) -> NDArray[ np.int64 ]:
    '''Return the next element without removing it.

    If the queue is empty, returns an empty int64 array (len = 0).
    '''
    if ( not self.data ): return np.empty( 0, dtype = np.int64 ) # R[1/2] queue is empty
    return self.data[ 0 ] # R[2/2]
