from collections import deque

from typing import Union, Sequence
from numpy.typing import NDArray
from numpy import int64

class Queue:
  '''Simple, unbounded FIFO queue being essentially a wrapper around the deque class.
  Heavily modified source from the origianl python 3.10 implementation. Eliminated unwanted operations and added peek()
  https://github.com/python/cpython/blob/3.10/Lib/queue.py
  '''

  def __init__( self ) -> None:
    self.data: deque[ NDArray[ int64 ] ] = deque()
    self.count: int = 0

  def put( self, item: NDArray[ int64 ] ) -> None:
    '''Put the item on the queue.'''
    self.data.append( item )
    self.count += 1 # increment the length counter

  def get( self ) -> NDArray[ int64 ]:
    '''Remove and return an item from the queue '''
    if ( self.count < 1 ): raise AssertionError( "The Queue is empty" )
    self.count -= 1 # increment the length counter
    return ( self.data.popleft() )

  def is_empty( self ) -> bool:
    '''Return True if the queue is empty, False otherwise.'''
    return ( self.count == 0 )
  
  def clear( self ) -> None:
    '''Clear the queue.'''
    self.data.clear()
    self.count = 0

  def size( self ) -> tuple[ int ]:
    '''Return the size of the queue.'''
    return ( self.count, )
  
  def peek( self ) -> Union[ Sequence[ int ], NDArray[ int64 ] ]:
    '''Returns the next element without popping it, returns an empty array if empty rather than throw an error'''
    if( self.count == 0 ): return ( [] ) # Empty 
    else: return ( self.data[0] ) # just acces leftmost element, don't pop it