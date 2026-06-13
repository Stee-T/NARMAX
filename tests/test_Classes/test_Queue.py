import unittest
import numpy as np

from NARMAX.Classes.Queue import Queue, QueueEmptyError

class TestQueue( unittest.TestCase ):
  def setUp( self ) -> None:
    '''Set up test fixtures before each test.'''
    self.queue = Queue()
    self.arr1 = np.array( [ 1, 2, 3 ], dtype = np.int64 )
    self.arr2 = np.array( [ 4, 5 ], dtype = np.int64 )

  # ------------------------------------------------------------------
  # Peek tests (existing, strengthened)
  # ------------------------------------------------------------------

  def test_peek_nonempty( self ) -> None:
    '''Peeking a non-empty queue returns the front item without removing it.'''
    self.queue.put( self.arr1 )
    self.assertEqual( self.queue.size(), 1 )
    peeked = self.queue.peek()
    self.assertIsInstance( peeked, np.ndarray )
    self.assertEqual( peeked.dtype, np.int64 )
    np.testing.assert_array_equal( peeked, self.arr1 )
    self.assertEqual( self.queue.size(), 1 )
    self.assertFalse( self.queue.is_empty() )
    np.testing.assert_array_equal( self.queue.get(), self.arr1 )

  def test_peek_empty( self ) -> None:
    '''Peeking an empty queue returns an empty numpy array, not an exception.'''
    self.assertTrue( self.queue.is_empty() )
    result = self.queue.peek()
    self.assertIsInstance( result, np.ndarray )
    self.assertEqual( result.dtype, np.int64 )
    self.assertEqual( len( result ), 0 )
    self.assertEqual( self.queue.size(), 0 )

  def test_peek_after_partial_get( self ) -> None:
    '''Peeking after a partial get still returns the next item.'''
    self.queue.put( self.arr1 )
    self.queue.put( self.arr2 )
    self.assertEqual( self.queue.size(), 2 )
    self.queue.get() # remove arr1
    self.assertEqual( self.queue.size(), 1 )
    peeked = self.queue.peek()
    self.assertIsInstance( peeked, np.ndarray )
    self.assertEqual( peeked.dtype, np.int64 )
    np.testing.assert_array_equal( peeked, self.arr2 )
    self.assertEqual( self.queue.size(), 1 )

  def test_peek_does_not_remove_item( self ) -> None:
    '''Peek should not change the queue size.'''
    self.queue.put( self.arr1 )
    self.assertEqual( self.queue.size(), 1 )
    _ = self.queue.peek()
    self.assertEqual( self.queue.size(), 1 )
    _ = self.queue.peek()
    self.assertEqual( self.queue.size(), 1 )
    self.assertFalse( self.queue.is_empty() )

  def test_peek_empty_returns_empty_array( self ) -> None:
    '''After emptying the queue, peek returns an empty array.'''
    self.queue.put( self.arr1 )
    self.queue.get() # now empty
    self.assertTrue( self.queue.is_empty() )
    peeked = self.queue.peek()
    self.assertIsInstance( peeked, np.ndarray )
    self.assertEqual( peeked.dtype, np.int64 )
    self.assertEqual( len( peeked ), 0 )
    np.testing.assert_array_equal( peeked, np.empty( 0, dtype = np.int64 ) )

  def test_len_of_peeked_empty_is_zero( self ) -> None:
    '''Length of peeked empty queue is zero – Arborescence depends on this.'''
    self.assertTrue( self.queue.is_empty() )
    result = self.queue.peek()
    self.assertIsInstance( result, np.ndarray )
    self.assertEqual( result.dtype, np.int64 )
    self.assertEqual( len( result ), 0 )

  # ------------------------------------------------------------------
  # QueueEmptyError
  # ------------------------------------------------------------------

  def test_queue_empty_error_is_exception( self ) -> None:
    '''QueueEmptyError is a subclass of Exception.'''
    self.assertTrue( issubclass( QueueEmptyError, Exception ) )

  def test_get_raises_queue_empty_error( self ) -> None:
    '''Getting from an empty queue raises QueueEmptyError with correct message.'''
    with self.assertRaises( QueueEmptyError ) as ctx:
      self.queue.get()
    self.assertEqual( str( ctx.exception ), "The Queue is empty" )

  # ------------------------------------------------------------------
  # get() – FIFO ordering
  # ------------------------------------------------------------------

  def test_get_returns_items_in_fifo_order( self ) -> None:
    '''Items are returned from the queue in FIFO order.'''
    self.queue.put( self.arr1 )
    self.queue.put( self.arr2 )
    self.assertEqual( self.queue.size(), 2 )
    got1 = self.queue.get()
    np.testing.assert_array_equal( got1, self.arr1 )
    self.assertEqual( self.queue.size(), 1 )
    got2 = self.queue.get()
    np.testing.assert_array_equal( got2, self.arr2 )
    self.assertEqual( self.queue.size(), 0 )
    self.assertTrue( self.queue.is_empty() )

  # ------------------------------------------------------------------
  # put() – type validation
  # ------------------------------------------------------------------

  def test_put_rejects_non_array( self ) -> None:
    '''put() raises TypeError for non-array items.'''
    for bad in [ None, 42, "string", [ 1, 2, 3 ], ( 1, 2 ), { "a": 1 } ]:
      with self.subTest( bad = bad ):
        with self.assertRaises( TypeError ) as ctx:
          self.queue.put( bad )  # type: ignore[arg-type]
        self.assertEqual( str( ctx.exception ), "Only integer numpy arrays are allowed" )
    self.assertEqual( self.queue.size(), 0 )

  def test_put_rejects_non_integer_array( self ) -> None:
    '''put() raises TypeError for non-integer numpy arrays.'''
    for dt in [ np.float64, np.float32, np.complex128, np.bool_ ]:
      with self.subTest( dtype = dt ):
        bad_arr = np.array( [ 1.0 ], dtype = dt )
        with self.assertRaises( TypeError ) as ctx:
          self.queue.put( bad_arr )
        self.assertEqual( str( ctx.exception ), "Only integer numpy arrays are allowed" )
    self.assertEqual( self.queue.size(), 0 )

  # ------------------------------------------------------------------
  # clear()
  # ------------------------------------------------------------------

  def test_clear_empties_queue( self ) -> None:
    '''clear() removes all items from the queue.'''
    self.queue.put( self.arr1 )
    self.queue.put( self.arr2 )
    self.assertEqual( self.queue.size(), 2 )
    self.assertFalse( self.queue.is_empty() )
    self.queue.clear()
    self.assertEqual( self.queue.size(), 0 )
    self.assertTrue( self.queue.is_empty() )
    np.testing.assert_array_equal( self.queue.peek(), np.empty( 0, dtype = np.int64 ) )

  def test_clear_on_empty_queue_does_nothing( self ) -> None:
    '''clear() on an already empty queue is a no-op.'''
    self.assertTrue( self.queue.is_empty() )
    self.queue.clear()
    self.assertTrue( self.queue.is_empty() )
    self.assertEqual( self.queue.size(), 0 )

  # ------------------------------------------------------------------
  # is_empty()
  # ------------------------------------------------------------------

  def test_is_empty_on_new_queue( self ) -> None:
    '''A newly created queue reports as empty.'''
    self.assertTrue( self.queue.is_empty() )
    self.assertEqual( self.queue.size(), 0 )

  def test_is_empty_after_put_and_get( self ) -> None:
    '''is_empty() transitions correctly through put/get cycles.'''
    self.assertTrue( self.queue.is_empty() )
    self.queue.put( self.arr1 )
    self.assertFalse( self.queue.is_empty() )
    self.queue.get()
    self.assertTrue( self.queue.is_empty() )

  # ------------------------------------------------------------------
  # size()
  # ------------------------------------------------------------------

  def test_size_after_multiple_operations( self ) -> None:
    '''size() returns the correct count through a sequence of operations.'''
    self.assertEqual( self.queue.size(), 0 )
    self.queue.put( self.arr1 )
    self.assertEqual( self.queue.size(), 1 )
    self.queue.put( self.arr2 )
    self.assertEqual( self.queue.size(), 2 )
    self.queue.put( self.arr1 )
    self.assertEqual( self.queue.size(), 3 )
    self.queue.get()
    self.assertEqual( self.queue.size(), 2 )
    self.queue.get()
    self.assertEqual( self.queue.size(), 1 )
    self.queue.get()
    self.assertEqual( self.queue.size(), 0 )
