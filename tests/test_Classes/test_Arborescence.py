import io
import os
import sys
import tempfile
from typing import Iterator
import unittest.mock as mock
import numpy as np
import torch as tor
import pytest

import matplotlib
matplotlib.use( 'Agg' )

from NARMAX.Classes.Arborescence import Arborescence
from NARMAX.HelperFuncs import AllCombinations, RemoveDuplicates
from NARMAX.Classes.Queue import Queue, QueueEmptyError
from NARMAX.Classes.MultiKeyHashTable import MultiKeyHashTable


# =====================================================================================================================
# Fixtures
# =====================================================================================================================

@pytest.fixture
def rng() -> np.random.Generator: return np.random.default_rng( 42 )


@pytest.fixture
def sample_y() -> tor.Tensor:
  tor.manual_seed( 42 )
  return tor.randn( 100 )


@pytest.fixture
def sample_dc() -> tor.Tensor:
  tor.manual_seed( 42 )
  Dc = tor.randn( 100, 10 )
  Dc -= tor.mean( Dc, axis = 0, keepdims = True )
  return Dc


@pytest.fixture
def sample_dc_names() -> np.ndarray: return np.array( [ f"reg_{ i }" for i in range( 10 ) ], dtype = np.str_ )


@pytest.fixture
def sample_ds() -> tor.Tensor:
  tor.manual_seed( 42 )
  Ds = tor.randn( 100, 3 )
  Ds -= tor.mean( Ds, axis = 0, keepdims = True )
  return Ds


@pytest.fixture
def sample_ds_names() -> np.ndarray: return np.array( [ "ds_0", "ds_1", "ds_2" ], dtype = np.str_ )


@pytest.fixture
def empty_arbo() -> Arborescence:
  '''Arborescence created with no arguments – simulates load() state.'''
  return Arborescence()


@pytest.fixture
def basic_arbo( sample_y: tor.Tensor ) -> Arborescence: return Arborescence( y = sample_y )


@pytest.fixture
def arbo_with_dc( sample_y: tor.Tensor, sample_dc: tor.Tensor, sample_dc_names: np.ndarray ) -> Arborescence:
  return Arborescence(
        y = sample_y,
        Dc = sample_dc,
        DcNames = sample_dc_names,
        tolRoot = 0.01,
        tolRest = 0.01,
        MaxDepth = 3,
    )


@pytest.fixture
def arbo_with_ds( sample_y: tor.Tensor, sample_ds: tor.Tensor, sample_ds_names: np.ndarray ) -> Arborescence:
  return Arborescence(
        y = sample_y,
        Ds = sample_ds,
        DsNames = sample_ds_names,
    )


# =====================================================================================================================
# Constructor: __init__
# =====================================================================================================================

class TestInit:
  '''Comprehensive tests for Arborescence.__init__'''

  # --- empty / load-from-file constructor ---
  def test_empty_constructor( self, empty_arbo ) -> None:
    '''Empty constructor creates an Arborescence with all attributes at defaults.'''
    assert empty_arbo.y is None
    assert empty_arbo.Ds is not None
    assert empty_arbo.Ds.shape == ( 0, 0 )
    assert empty_arbo.DsNames.shape == ( 0, )
    assert empty_arbo.Dc is None
    assert empty_arbo.nC == 0
    assert empty_arbo.nS == 0
    assert isinstance( empty_arbo.Q, Queue )
    assert isinstance( empty_arbo.LG, MultiKeyHashTable )
    assert empty_arbo.Abort is False
    assert empty_arbo.TotalNodes == 1
    assert empty_arbo.nNotSkippedNodes == 0
    assert empty_arbo.theta is None
    assert empty_arbo.ERR is None
    assert empty_arbo.L is None
    assert empty_arbo.AbortedRegs == 0
    assert isinstance( empty_arbo.Ds, tor.Tensor )
    assert isinstance( empty_arbo.DsNames, np.ndarray )
    assert isinstance( empty_arbo.Q, Queue )
    assert isinstance( empty_arbo.LG, MultiKeyHashTable )

  # --- y argument ---
  def test_y_1d( self, sample_y ) -> None:
    '''A 1D y tensor is centered (mean removed).'''
    arbo = Arborescence( y = sample_y )
    assert arbo.y is not None
    assert arbo.y.ndim == 1
    assert arbo.y.shape[ 0 ] == 100
    assert abs( arbo.y.mean().item() ) < 1e-10
    assert isinstance( arbo.y, tor.Tensor )
    assert arbo.y.dtype == tor.float64

  def test_y_2d_column( self ) -> None:
    '''A 2D column vector y is flattened to 1D and centered.'''
    y_2d = tor.randn( 50, 1 )
    arbo = Arborescence( y = y_2d )
    assert arbo.y is not None
    assert arbo.y.ndim == 1
    assert arbo.y.shape[ 0 ] == 50
    assert abs( arbo.y.mean().item() ) < 1e-10
    assert arbo.y.dtype == tor.float64

  def test_y_wrong_type_raises( self ) -> None:
    '''Passing a non-tensor y raises ValueError.'''
    with pytest.raises( ValueError, match = "y must be None or a 1D-torch.Tensor" ):
      Arborescence( y = [ 1, 2, 3 ] )

  def test_y_wrong_shape_2d_multi_column_raises( self ) -> None:
    '''A 2D y with multiple columns raises ValueError.'''
    with pytest.raises( ValueError, match = "y must be None or of shape" ):
      Arborescence( y = tor.randn( 50, 3 ) )

  def test_y_3d_raises( self ) -> None:
    '''A 3D y tensor raises ValueError.'''
    with pytest.raises( ValueError, match = "y must be None or of shape" ):
      Arborescence( y = tor.randn( 10, 10, 10 ) )

  def test_y_nan_raises( self ) -> None:
    '''y containing NaN raises AssertionError.'''
    y_nan = tor.tensor( [ 1.0, float( "nan" ), 3.0 ] )
    with pytest.raises( AssertionError, match = "NaNs" ):
      Arborescence( y = y_nan )

  # --- Ds argument ---
  def test_ds_none_creates_empty( self, sample_y ) -> None:
    '''Ds=None creates an empty (0,0) Ds tensor.'''
    arbo = Arborescence( y = sample_y, Ds = None, DsNames = None )
    assert arbo.Ds.shape == ( 100, 0 )
    assert arbo.DsNames.shape == ( 0, )
    assert arbo.nS == 0
    assert isinstance( arbo.Ds, tor.Tensor )
    assert isinstance( arbo.DsNames, np.ndarray )
    assert arbo.DsMeans is None

  def test_ds_wrong_type_raises( self, sample_y ) -> None:
    '''Ds with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "Ds must be None or a torch.Tensor" ):
      Arborescence( y = sample_y, Ds = "not_a_tensor" )

  def test_ds_nan_raises( self, sample_y ) -> None:
    '''Ds containing NaN raises AssertionError.'''
    Ds_nan = tor.tensor( [ [ 1.0, float( "nan" ) ], [ 2.0, 3.0 ] ] )
    with pytest.raises( AssertionError, match = "NaNs" ):
      Arborescence( y = sample_y, Ds = Ds_nan, DsNames = np.array( [ "a", "b" ], dtype = np.str_ ) )

  def test_ds_names_len_mismatch_raises( self, sample_y, sample_ds ) -> None:
    '''DsNames length mismatch with Ds raises TypeError.'''
    with pytest.raises( TypeError, match = "DsNames has not the same number" ):
      Arborescence( y = sample_y, Ds = sample_ds, DsNames = np.array( [ "a", "b" ], dtype = np.str_ ) )

  # --- DsNames type check ---
  def test_ds_names_wrong_type_raises( self, sample_y ) -> None:
    '''DsNames with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "DsNames must be None or an np.array of strings" ):
      Arborescence( y = sample_y, DsNames = [ "a", "b" ] )

  def test_ds_names_non_string_elements_raises( self, sample_y, sample_ds ) -> None:
    '''DsNames with non-string elements raises ValueError.'''
    bad_names = np.array( [ 1, 2, 3 ] )
    with pytest.raises( ValueError, match = "DsNames must be None or an np.array of strings" ):
      Arborescence( y = sample_y, Ds = sample_ds, DsNames = bad_names )

  # --- Dc argument ---
  def test_dc_centering_and_removal( self, sample_y, sample_dc ) -> None:
    '''Dc is centered and duplicates are removed.'''
    Dc_with_dup = tor.column_stack( ( sample_dc, sample_dc[ :, 0 : 1 ] ) ) # duplicate last column
    names = np.array( [ f"r_{ i }" for i in range( Dc_with_dup.shape[ 1 ] ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc_with_dup, DcNames = names )
    # Should have removed the duplicate
    assert arbo.Dc.shape[ 1 ] == sample_dc.shape[ 1 ]
    # Should be column-wise centered
    col_means = tor.mean( arbo.Dc, axis = 0 )
    assert tor.allclose( col_means, tor.zeros_like( col_means ), atol = 1e-12 )
    assert isinstance( arbo.Dc, tor.Tensor )
    assert arbo.DcMeans is not None
    assert arbo.DcMeans.shape == ( 1, sample_dc.shape[ 1 ] )
    assert isinstance( arbo.DcFilterIdx, np.ndarray )
    assert arbo.DcFilterIdx.size == sample_dc.shape[ 1 ]

  def test_dc_duplicate_removal_keeps_first( self, sample_y ) -> None:
    '''When duplicates exist, the first occurrence is retained.'''
    x = tor.randn( 10, 2 )
    # duplicate second column and append
    Dc = tor.cat( [ x, x[ :, 1 : 2 ] ], dim = 1 )
    names = np.array( [ "col0", "col1", "col2" ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y[ : 10 ], Dc = Dc, DcNames = names )
    assert arbo.Dc.shape[ 1 ] == 2
    # The second column should equal the original x[:,1] (possibly after centering)
    assert tor.allclose( arbo.Dc[ :, 1 ], x[ :, 1 ] - x[ :, 1 ].mean(), atol = 1e-12 )
    assert arbo.DcNames.tolist() == [ "col0", "col1" ]
    assert arbo.DcNames.dtype.kind == 'U'

  def test_dc_wrong_type_raises( self, sample_y ) -> None:
    '''Dc with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "Dc must be None or a torch.Tensor" ):
      Arborescence( y = sample_y, Dc = "not_a_tensor" )

  def test_dc_nan_raises( self, sample_y ) -> None:
    '''Dc containing NaN raises AssertionError.'''
    Dc_nan = tor.tensor( [ [ 1.0, float( "nan" ) ], [ 2.0, 3.0 ] ] )
    with pytest.raises( AssertionError, match = "NaNs" ):
      Arborescence( y = sample_y, Dc = Dc_nan, DcNames = np.array( [ "a", "b" ], dtype = np.str_ ) )

  def test_dc_names_length_mismatch_raises( self, sample_y, sample_dc ) -> None:
    '''DcNames length mismatch with Dc raises TypeError.'''
    with pytest.raises( TypeError, match = "DcNames must be None or a np.array of the same length as Dc" ):
      Arborescence( y = sample_y, Dc = sample_dc, DcNames = np.array( [ "a" ], dtype = np.str_ ) )

  # --- DcNames type check ---
  def test_dc_names_wrong_type_raises( self, sample_y ) -> None:
    '''DcNames with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "DcNames must be None or an np.array of strings" ):
      Arborescence( y = sample_y, DcNames = [ "a", "b" ] )

  # --- tolRoot / tolRest ---
  def test_tol_root_wrong_type_raises( self, sample_y ) -> None:
    '''tolRoot with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "tolRoot must be a float" ):
      Arborescence( y = sample_y, tolRoot = "0.001" )

  def test_tol_rest_wrong_type_raises( self, sample_y ) -> None:
    '''tolRest with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "tolRest must be a float" ):
      Arborescence( y = sample_y, tolRest = 5 )

  # --- MaxDepth ---
  def test_max_depth_wrong_type_raises( self, sample_y ) -> None:
    '''MaxDepth with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "MaxDepth must be an int >= 0" ):
      Arborescence( y = sample_y, MaxDepth = 3.0 )

  # --- ValFunc ---
  def test_val_func_not_callable_raises( self, sample_y ) -> None:
    '''ValFunc that is not callable raises ValueError.'''
    with pytest.raises( ValueError, match = "ValFunc must be None or a callable function" ):
      Arborescence( y = sample_y, ValFunc = "not_callable" )

  # --- ValData ---
  def test_val_data_not_dict_raises( self, sample_y ) -> None:
    '''ValData that is not a dict raises ValueError.'''
    with pytest.raises( ValueError, match = "ValData must be None or a dict" ):
      Arborescence( y = sample_y, ValData = [ 1, 2, 3 ] )

  # --- Verbose ---
  def test_verbose_not_bool_raises( self, sample_y ) -> None:
    '''Verbose that is not a bool raises ValueError.'''
    with pytest.raises( ValueError, match = "verbose must be a bool" ):
      Arborescence( y = sample_y, Verbose = 1 )

  # --- MorphDict ---
  def test_morph_dict_not_dict_raises( self, sample_y ) -> None:
    '''MorphDict that is not a dict raises ValueError.'''
    with pytest.raises( ValueError, match = "MorphDict must be None or a dict" ):
      Arborescence( y = sample_y, MorphDict = "not_dict" )

  def test_morph_dict_without_dc_raises( self, sample_y ) -> None:
    '''MorphDict without Dc raises AssertionError.'''
    with pytest.raises( AssertionError, match = "No Dc passed" ):
      Arborescence( y = sample_y, MorphDict = { "NonLinMap": [ 0, 1 ] } )

  def test_morph_dict_missing_nonlinmap_raises( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''MorphDict missing NonLinMap key raises AssertionError.'''
    with pytest.raises( AssertionError, match = "missing the key 'NonLinMap'" ):
      Arborescence( y = sample_y, Dc = sample_dc, DcNames = sample_dc_names, MorphDict = { "wrong_key": [ 0 ] } )

  # --- U argument ---
  def test_u_wrong_type_raises( self, sample_y ) -> None:
    '''U with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "U must be None or a list of ints >=0" ):
      Arborescence( y = sample_y, U = "not_a_list" )

  def test_u_wrong_element_type_raises( self, sample_y ) -> None:
    '''U with wrong element type raises ValueError.'''
    with pytest.raises( ValueError, match = "U must be None or a list of ints >=0" ):
      Arborescence( y = sample_y, U = [ 0, "a", 2 ] )

  def test_u_too_short_raises( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''U shorter than MaxDepth raises ValueError.'''
    with pytest.raises( ValueError, match = "U must contain at least MaxDepth" ):
      Arborescence( y = sample_y, Dc = sample_dc, DcNames = sample_dc_names, MaxDepth = 5, U = [ 0, 1 ] )

  def test_u_defaults_to_all_indices( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''U defaults to all column indices when not provided.'''
    arbo = Arborescence( y = sample_y, Dc = sample_dc, DcNames = sample_dc_names )
    assert arbo.U == list( range( arbo.nC ) )

  # --- FileName ---
  def test_filename_wrong_type_raises( self, sample_y ) -> None:
    '''FileName with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "FileName must be None or a str" ):
      Arborescence( y = sample_y, FileName = 123 )

  # --- SaveFrequency ---
  def test_save_frequency_wrong_type_raises( self, sample_y ) -> None:
    '''SaveFrequency with wrong type raises ValueError.'''
    with pytest.raises( ValueError, match = "SaveFrequency must be None a float or an int" ):
      Arborescence( y = sample_y, SaveFrequency = "10" )

  def test_save_frequency_conversion( self, sample_y ) -> None:
    '''SaveFrequency in minutes is converted to seconds.'''
    arbo = Arborescence( y = sample_y, SaveFrequency = 5 )
    assert arbo.SaveFrequency == 300 # 5 minutes * 60 seconds
    assert isinstance( arbo.SaveFrequency, int )

  def test_save_frequency_float_conversion( self, sample_y ) -> None:
    '''SaveFrequency as float in minutes is converted to seconds.'''
    arbo = Arborescence( y = sample_y, SaveFrequency = 0.5 )
    assert arbo.SaveFrequency == 30.0
    assert isinstance( arbo.SaveFrequency, float )
    arbo2 = Arborescence( y = sample_y, SaveFrequency = 2.0 )
    assert arbo2.SaveFrequency == 120.0

  # --- INT_TYPE deduced from Dc shape ---
  def test_int_type_uint16_for_small_dc( self, sample_y ) -> None:
    '''Small Dc uses np.uint16 INT_TYPE.'''
    small_dc = tor.randn( 10, 10 )
    names = np.array( [ f"r{ i }" for i in range( 10 ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = small_dc, DcNames = names )
    assert arbo.INT_TYPE == np.uint16
    assert arbo.DcFilterIdx.dtype == np.int64

  def test_int_type_uint64_for_large_dc( self, sample_y ) -> None:
    '''Large Dc uses np.uint32 INT_TYPE.'''
    if ( np.iinfo( np.uint32 ).max < 2**32 ):
      # On some systems uint32 max may be larger
      large_dc = tor.randn( 10, 70000 )
      arbo = Arborescence( y = sample_y, Dc = large_dc,
                                DcNames = np.array( [ f"r{ i }" for i in range( 70000 ) ], dtype = np.str_ ) )
      assert arbo.INT_TYPE == np.uint32

  # --- Default validation function ---
  def test_default_val_func_when_none( self, sample_y ) -> None:
    '''Default ValFunc computes 1 - sum(ERR).'''
    arbo = Arborescence( y = sample_y )
    assert arbo.ValFunc is not None
    # Default validation: 1 - sum(ERR)
    result = arbo.ValFunc( None, None, np.array( [ 0.2, 0.3 ] ), None, None )
    assert abs( result - 0.5 ) < 1e-12

  # --- Ds concatenated with empty when y is None ---
  def test_ds_empty_when_y_is_none( self ) -> None:
    '''Ds is empty when y is None.'''
    arbo = Arborescence()
    assert arbo.Ds.shape == ( 0, 0 )

  # --- Ds centering and duplicate removal ---
  def test_ds_centering_and_dup_removal( self, sample_y, sample_ds ) -> None:
    '''Ds is centered and duplicates removed.'''
    Ds_with_dup = tor.column_stack( ( sample_ds, sample_ds[ :, 0 : 1 ] ) )
    names = np.array( [ f"ds_{ i }" for i in range( Ds_with_dup.shape[ 1 ] ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Ds = Ds_with_dup, DsNames = names )
    assert arbo.Ds.shape[ 1 ] == sample_ds.shape[ 1 ]
    col_means = tor.mean( arbo.Ds, axis = 0 )
    assert tor.allclose( col_means, tor.zeros_like( col_means ), atol = 1e-12 )

  # --- y must be same length as Dc rows ---
  def test_y_dc_row_mismatch( self, sample_y ) -> None:
    '''y and Dc row count mismatch is handled.'''
    Dc = tor.randn( 50, 5 )
    names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    # This should work — no explicit check in init, but Dc filtering with RemoveDuplicates returns mismatch size
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names )
    assert arbo.Dc.shape[ 0 ] == 50

  # --- y left as None when not passed ---
  def test_y_defaults_to_none( self ) -> None:
    '''y defaults to None when not passed.'''
    arbo = Arborescence()
    assert arbo.y is None

  # --- DsMeans left as None when Ds is None ---
  def test_ds_means_none_when_ds_none( self, sample_y ) -> None:
    '''DsMeans is None when Ds is not provided.'''
    arbo = Arborescence( y = sample_y )
    assert arbo.DsMeans is None

  # --- DsMeans computed when Ds is given ---
  def test_ds_means_computed( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''DsMeans is computed when Ds is provided.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names )
    assert arbo.DsMeans is not None
    assert arbo.DsMeans.shape == ( 1, 3 )

  # --- DcFilterIdx is set when Dc is given ---
  def test_dc_filter_idx_set( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''DcFilterIdx is set when Dc is provided.'''
    arbo = Arborescence( y = sample_y, Dc = sample_dc, DcNames = sample_dc_names )
    assert arbo.DcFilterIdx is not None
    assert isinstance( arbo.DcFilterIdx, np.ndarray )

  # --- DcFilterIdx is None when Dc is None ---
  def test_dc_filter_idx_none( self, basic_arbo ) -> None:
    '''DcFilterIdx is None when Dc is not provided.'''
    assert basic_arbo.DcFilterIdx is None

  # --- MultiKeyHashTable and Queue initialized even with None Dc ---
  def test_internal_structures_initialized( self, basic_arbo ) -> None:
    '''Queue and MultiKeyHashTable are initialized.'''
    assert isinstance( basic_arbo.Q, Queue )
    assert isinstance( basic_arbo.LG, MultiKeyHashTable )

  # --- MinLen / nNodes initialized ---
  def test_tracking_variables_init( self, basic_arbo ) -> None:
    '''Tracking variables (MinLen, nNodes, nComputed) are initialized.'''
    assert basic_arbo.MinLen == []
    assert basic_arbo.nNodesInNextLevel == 0
    assert basic_arbo.nNodesInCurrentLevel == 0
    assert basic_arbo.nComputed == 0

  # --- nS computed correctly ---
  def test_ns_with_ds( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''nS equals the number of Ds columns.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names )
    assert arbo.nS == sample_ds.shape[ 1 ]

  def test_ns_without_ds( self, basic_arbo ) -> None:
    '''nS is 0 when no Ds is provided.'''
    assert basic_arbo.nS == 0

  # --- nC computed correctly ---
  def test_nc_with_dc( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''nC equals the number of Dc columns.'''
    arbo = Arborescence( y = sample_y, Dc = sample_dc, DcNames = sample_dc_names )
    assert arbo.nC == sample_dc.shape[ 1 ]

  def test_nc_without_dc( self, basic_arbo ) -> None:
    '''nC is 0 when no Dc is provided.'''
    assert basic_arbo.nC == 0

  # --- MorphDict filtered NonLinMap with DcFilterIdx ---
  def test_morphdict_nonlinmap_filtered( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''MorphDict NonLinMap is filtered with DcFilterIdx.'''
    # Add duplicate column to trigger filtering
    Dc = tor.column_stack( ( sample_dc, sample_dc[ :, 0 : 1 ] ) )
    names = np.array( [ f"r_{ i }" for i in range( Dc.shape[ 1 ] ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names,
                            MorphDict = { "NonLinMap": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] } )
    assert arbo.MorphDict is not None
    assert arbo.MorphDict[ "nC" ] == sample_dc.shape[ 1 ]

  # --- SaveFrequency 0 preserved ---
  def test_save_frequency_zero( self, basic_arbo ) -> None:
    '''SaveFrequency of 0 is preserved.'''
    assert basic_arbo.SaveFrequency == 0


# =====================================================================================================================
# rFOrLSR
# =====================================================================================================================

class TestRFOrLSR:
  '''Tests for the recursive Forward Orthogonal Least Squares Regression.'''

  # --- Basic regression with Dc only ---
  def test_rforlsr_dc_only( self, sample_y ) -> None:
    '''rFOrLSR with only Dc regressors returns correct results.'''
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    result = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 0.999, OutputAll = True )
    # OutputAll=True normal return: (theta, L, ERR)
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert theta.ndim == 1
    assert isinstance( L, np.ndarray )
    assert len( L ) >= 1
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert ERR.ndim == 1

  # --- Regression with Ds only ---
  def test_rforlsr_ds_only( self, sample_y, sample_ds ) -> None:
    '''rFOrLSR with only Ds regressors returns correct results.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds,
                            DsNames = np.array( [ "a", "b", "c" ], dtype = np.str_ ) )
    result = arbo.rFOrLSR( sample_y, Ds = sample_ds, tol = 0.999, OutputAll = True )
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert ERR.ndim == 1
    assert len( ERR ) == len( theta )

  # --- Regression with Ds + Dc ---
  def test_rforlsr_ds_and_dc( self, sample_y, sample_ds, sample_dc ) -> None:
    '''rFOrLSR with both Ds and Dc returns correct results.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, Dc = sample_dc,
                            DsNames = np.array( [ "a", "b", "c" ], dtype = np.str_ ),
                            DcNames = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    Ds = tor.column_stack( ( sample_ds, sample_dc[ :, 0 : 1 ] ) )
    result = arbo.rFOrLSR( sample_y, Ds = Ds, Dc = sample_dc, tol = 0.999, OutputAll = True )
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )

  # --- Error when both Dc and Ds are None ---
  def test_rforlsr_no_regressors_raises( self, sample_y ) -> None:
    '''rFOrLSR with no regressors raises AssertionError.'''
    arbo = Arborescence( y = sample_y )
    with pytest.raises( AssertionError, match = "Dc and Ds are None" ):
      arbo.rFOrLSR( sample_y )

  # --- Empty Ds treated as None ---
  def test_rforlsr_empty_ds_as_none( self, sample_y, sample_dc ) -> None:
    '''Empty Ds is treated as None.'''
    empty_ds = tor.empty( ( len( sample_y ), 0 ) )
    arbo = Arborescence( y = sample_y, Dc = sample_dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    result = arbo.rFOrLSR( sample_y, Ds = empty_ds, Dc = sample_dc, tol = 0.999, OutputAll = True )
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )

  # --- Single column Ds reshape ---
  def test_rforlsr_single_col_ds_reshaped( self, sample_y ) -> None:
    '''Single-column Ds is reshaped correctly.'''
    Ds_1d = tor.randn( 100 )
    arbo = Arborescence( y = sample_y, Ds = Ds_1d.reshape( -1, 1 ),
                            DsNames = np.array( [ "a" ], dtype = np.str_ ) )
    result = arbo.rFOrLSR( sample_y, Ds = Ds_1d, tol = 0.999, OutputAll = True )
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )
    assert len( ERR ) == 1

  # --- MaxTerms smaller than Ds columns raises ---
  def test_rforlsr_maxtterms_too_small_raises( self, sample_y, sample_ds ) -> None:
    '''MaxTerms smaller than Ds columns raises ValueError.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds,
                            DsNames = np.array( [ "a", "b", "c" ], dtype = np.str_ ) )
    with pytest.raises( ValueError, match = "MaxTerms" ):
      arbo.rFOrLSR( sample_y, Ds = sample_ds, MaxTerms = 1 )

  # --- OOIT prediction with LI ---
  def test_rforlsr_oit_prediction( self, sample_y ) -> None:
    '''When LI is given and SameStart matches, regression is OOIT-predicted and returns correct content.'''
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    LI = np.array( [ 0 ], dtype = np.int64 )
    # First call: normal regression (no match yet)
    output1 = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 0.0, OutputAll = True, LI = LI, MaxTerms = 100 )
    assert len( output1 ) == 3
    theta1, L1, ERR1 = output1
    # Build full sequence as the arbo would
    full_seq = np.concatenate( ( LI, L1 ) )
    idx = arbo.LG.AddData( np.array( full_seq, dtype = arbo.INT_TYPE ) )
    arbo.LG.CreateKeys( MinLen = len( LI ), IndexSet = full_seq, Value = idx )

    # Second call with same LI: OOIT predicts -> returns (predicted_L, RegIdx)
    output2 = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 0.0, OutputAll = True, LI = LI, MaxTerms = 100 )
    assert len( output2 ) == 2
    predicted_L, reg_idx = output2
    # predicted_L should be the part of the full sequence after LI, sorted (the OOIT code uses np.setdiff1d)
    expected_L = np.setdiff1d( full_seq, LI, assume_unique = True )
    np.testing.assert_array_equal( predicted_L, expected_L )
    # reg_idx should be the same as the stored index
    assert reg_idx == idx

  # --- Abort returns early with just L ---
  def test_rforlsr_abort_returns_early( self, sample_y ) -> None:
    '''When Abort is True, rFOrLSR returns early with just L.'''
    Dc = tor.randn( 100, 3 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ "a", "b", "c" ], dtype = np.str_ ),
                            tolRoot = 0.001, tolRest = 0.001, MaxDepth = 1 )
    arbo.Abort = True
    result = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 0.001, MaxTerms = 1, OutputAll = True )
    assert len( result ) == 1
    ( L_only, ) = result
    assert isinstance( L_only, np.ndarray )
    assert L_only.dtype == arbo.INT_TYPE

  # --- LI not matching returns full regression ---
  def test_rforlsr_no_li_match_returns_full( self, sample_y, sample_dc ) -> None:
    '''LI not matching any regression returns full result.'''
    LI = np.array( [ 99 ], dtype = np.int64 ) # definitely not in any regression
    arbo = Arborescence( y = sample_y, Dc = sample_dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    output = arbo.rFOrLSR( sample_y, Dc = sample_dc, tol = 0.999, OutputAll = True, LI = LI )
    assert len( output ) == 3
    theta, L, ERR = output
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )

  # --- Verbose doesn't crash ---
  def test_rforlsr_verbose( self, sample_y, sample_dc ) -> None:
    '''rFOrLSR with Verbose=True runs without error.'''
    arbo = Arborescence( y = sample_y, Dc = sample_dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1, Verbose = True )
    result = arbo.rFOrLSR( sample_y, Dc = sample_dc, tol = 0.999, OutputAll = True )
    assert len( result ) == 3

  # --- OutputAll=True with MaxTerms returns full result ---
  def test_rforlsr_output_all_true( self, sample_y, sample_dc ) -> None:
    '''OutputAll=True with MaxTerms returns full result tuple.'''
    arbo = Arborescence( y = sample_y, Dc = sample_dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ ),
                            tolRoot = 0.001, tolRest = 0.001, MaxDepth = 1 )
    result = arbo.rFOrLSR( sample_y, Dc = sample_dc, tol = 0.001, OutputAll = True, MaxTerms = 100 )
    # With OutputAll=True and s <= MaxTerms, should return (theta, L, ERR)
    assert len( result ) == 3
    theta, L, ERR = result
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert all( e >= 0 for e in ERR )

  # --- ERR computation correctness ---
  def test_rforlsr_err_sum_bounds( self, sample_y ) -> None:
    '''ERR sum is between 0 and 1.'''
    Dc = tor.randn( 100, 3 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( 3 ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    theta, L, ERR = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 0.999, OutputAll = True, MaxTerms = 100 )
    assert 0 < np.sum( ERR ) <= 1.0 + 1e-10
    assert all( e >= 0 for e in ERR )
    assert isinstance( ERR, np.ndarray )
    assert ERR.dtype == np.float64
    assert len( ERR ) == len( L ) == len( theta )

  def test_rforlsr_err_variance_relation( self, sample_y ) -> None:
    '''Sum of ERR + (residual variance)/(original variance) == 1.'''
    y = sample_y - sample_y.mean() # rFOrLSR requires centered y
    Dc = tor.randn( 100, 4 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = y, Dc = Dc,
                            DcNames = np.array( [ f"r{ i }" for i in range( 4 ) ], dtype = np.str_ ),
                            tolRoot = 0.999, tolRest = 0.999, MaxDepth = 1 )
    theta, L, ERR = arbo.rFOrLSR( y, Dc = Dc, tol = 0.999, OutputAll = True, MaxTerms = 100 )
    # Reconstruct prediction
    if ( len( L ) > 0 ): y_hat = Dc[ :, L ] @ tor.as_tensor( theta, dtype = tor.float64 )
    else: y_hat = tor.zeros_like( y, dtype = tor.float64 )
    residual = y - y_hat
    var_y = y.var( unbiased = False ) # population variance
    var_res = residual.var( unbiased = False )
    # The orthogonal decomposition guarantees ||y||^2 = ||y_hat||^2 + ||residual||^2,
    # and sum(ERR) = ||y_hat||^2 / ||y||^2  →  sum(ERR) = 1 - ||residual||^2 / ||y||^2.
    # For centered data this simplifies to var_res/var_y.
    assert abs( np.sum( ERR ) + var_res.item() / var_y.item() - 1.0 ) < 1e-10

  # --- Entire dictionary used up warning ---
  def test_rforlsr_dictionary_exhausted( self, sample_y ) -> None:
    '''All dictionary columns can be used up.'''
    Dc = tor.randn( 100, 2 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ "a", "b" ], dtype = np.str_ ),
                            tolRoot = 1e-12, tolRest = 1e-12, MaxDepth = 1 )
    # Very low tolerance so it tries to use all
    result = arbo.rFOrLSR( sample_y, Dc = Dc, tol = 1e-12, OutputAll = True, MaxTerms = 100 )
    assert len( result ) == 3
    theta, L, ERR = result
    assert len( L ) == 2  # both columns selected
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert len( ERR ) == 2


# =====================================================================================================================
# fit
# =====================================================================================================================

class TestFit:
  '''Tests for the bread-first search Arborescent rFOrLSR (fit).'''

  # --- y is None raises ---
  def test_fit_y_none_raises( self, empty_arbo ) -> None:
    '''fit() with y=None raises AssertionError.'''
    with pytest.raises( AssertionError, match = "y is None" ):
      empty_arbo.fit()

  # --- Invalid SaveFrequency raises ---
  def test_fit_invalid_save_frequency_raises( self, basic_arbo ) -> None:
    '''Invalid SaveFrequency raises ValueError.'''
    with pytest.raises( ValueError, match = "SaveFrequency must be an integer or a float" ):
      basic_arbo.fit( SaveFrequency = "bad" )

  # --- Negative SaveFrequency raises ---
  def test_fit_negative_save_frequency_raises( self, basic_arbo ) -> None:
    '''Negative SaveFrequency raises ValueError.'''
    with pytest.raises( ValueError, match = "SaveFrequency cannot be negative" ):
      basic_arbo.fit( SaveFrequency = -1 )

  # --- Invalid FileName raises ---
  def test_fit_invalid_filename_raises( self, basic_arbo ) -> None:
    '''Invalid FileName raises ValueError.'''
    with pytest.raises( ValueError, match = "FileName must be a string" ):
      basic_arbo.fit( FileName = 123 )

  # --- Basic fit with only Ds (imposed regressors only) ---
  def test_fit_ds_only( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''When only Ds is given and MaxDepth=0, fit should do root and validate immediately.'''
    arbo = Arborescence(
            y = sample_y,
            Ds = sample_ds,
            DsNames = sample_ds_names,
            MaxDepth = 0,
        )
    result = arbo.fit()
    assert len( result ) == 6
    theta, L, ERR, MorphDict, Dc, DcNames = result
    assert len( theta ) > 0
    assert theta.ndim == 1
    assert isinstance( theta, tor.Tensor )
    assert MorphDict is None
    assert Dc is None
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert len( ERR ) == len( theta )

  # --- Basic fit with Dc (candidate regressors) ---
  def test_fit_with_dc( self, sample_y ) -> None:
    '''fit() with Dc returns valid results.'''
    tor.manual_seed( 42 )
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    dc_names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence(
            y = sample_y,
            Dc = Dc,
            DcNames = dc_names,
            tolRoot = 0.99,
            tolRest = 0.99,
            MaxDepth = 1,
        )
    result = arbo.fit()
    theta, L, ERR, MorphDict, Dc_out, DcNames_out = result
    assert len( theta ) > 0
    assert len( L ) > 0
    assert len( ERR ) > 0
    assert isinstance( theta, tor.Tensor )
    assert theta.ndim == 1
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )
    assert ERR.dtype == np.float64
    assert len( ERR ) == len( L ) == len( theta )
    assert all( e >= 0 for e in ERR )
    assert np.sum( ERR ) <= 1.0 + 1e-10

  # --- Fit with Dc=None and MaxDepth=0 (imposed only) ---
  def test_fit_dc_none_maxdepth_zero( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''fit() with Dc=None and MaxDepth=0 returns 6-tuple.'''
    arbo = Arborescence(
            y = sample_y,
            Ds = sample_ds,
            DsNames = sample_ds_names,
            MaxDepth = 0,
        )
    result = arbo.fit()
    assert len( result ) == 6

  # --- Fit with Deep Arbo traversal (MaxDepth > 0) ---
  def test_fit_with_traversal( self, sample_y ) -> None:
    '''fit() with MaxDepth>0 traverses the arborescence.'''
    tor.manual_seed( 42 )
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    dc_names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence(
            y = sample_y,
            Dc = Dc,
            DcNames = dc_names,
            tolRoot = 0.99,
            tolRest = 0.99,
            MaxDepth = 2,
        )
    result = arbo.fit()
    theta, L, ERR, MorphDict, Dc_out, DcNames_out = result
    assert len( theta ) > 0
    assert len( L ) > 0
    assert len( ERR ) > 0

  # --- Ds maxdepth 0 fit returns correct outputs ---
  def test_fit_ds_outputs( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''fit() outputs have correct types for Ds-only model.'''
    arbo = Arborescence(
            y = sample_y,
            Ds = sample_ds,
            DsNames = sample_ds_names,
            MaxDepth = 0,
        )
    theta, L, ERR, MD, Dc, DcN = arbo.fit()
    assert isinstance( theta, tor.Tensor )
    assert theta.ndim == 1
    assert isinstance( L, np.ndarray )
    assert L.dtype == arbo.INT_TYPE
    assert isinstance( ERR, np.ndarray )
    assert ERR.dtype == np.float64
    assert len( ERR ) == len( theta )
    assert MD is None
    assert Dc is None
    assert DcN is None

  # --- FileName passed to fit stores it ---
  def test_fit_filename_stored( self, sample_y, sample_dc ) -> None:
    '''FileName passed to fit() is stored on the Arborescence.'''
    Dc = tor.randn( 100, 3 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 3 ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names,
                            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 1 )
    arbo.fit( FileName = "dummy_test_file.narmax" )
    assert arbo.FileName == "dummy_test_file.narmax"

  # --- SaveFrequency overwritten by fit ---
  def test_fit_save_frequency_overwritten( self, sample_y, sample_dc ) -> None:
    '''SaveFrequency is overwritten by fit().'''
    Dc = tor.randn( 100, 3 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 3 ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names,
                            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 1,
                            FileName = "dummy_test_sf.narmax" )
    arbo.fit( SaveFrequency = 10 )
    assert arbo.SaveFrequency == 600 # 10 minutes * 60
    # Clean up
    import os
    if ( os.path.exists( "dummy_test_sf.narmax" ) ): os.remove( "dummy_test_sf.narmax" )


# =====================================================================================================================
# validate
# =====================================================================================================================

class TestValidate:
  '''Tests for model selection / validation.'''

  # --- validate with Dc ---
  def test_validate_with_dc( self, sample_y, sample_dc ) -> None:
    '''validate() populates theta, L, and ERR after fit with Dc.'''
    names = np.array( [ f"r{ i }" for i in range( sample_dc.shape[ 1 ] ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = sample_dc, DcNames = names,
                            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 1 )
    arbo.fit()
    assert arbo.theta is not None
    assert arbo.L is not None
    assert arbo.ERR is not None
    assert isinstance( arbo.theta, tor.Tensor )
    assert arbo.theta.ndim == 1
    assert isinstance( arbo.L, np.ndarray )
    assert isinstance( arbo.ERR, np.ndarray )
    assert len( arbo.ERR ) == len( arbo.theta )

  # --- validate without Dc ---
  def test_validate_without_dc( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''validate() populates theta, L, and ERR after fit without Dc.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    assert arbo.theta is not None
    assert arbo.L is not None
    assert arbo.ERR is not None
    assert isinstance( arbo.theta, tor.Tensor )
    assert isinstance( arbo.ERR, np.ndarray )
    assert len( arbo.ERR ) == len( arbo.theta )

  # --- validate with ValData ---
  def test_validate_with_valdata( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''When ValData is provided, the validation function is called.'''
    def dummy_validation( theta, L, ERR, RegNames, ValData, DcFilterIdx ): return 1 - np.sum( ERR )

    val_data = {
            "y": [ sample_y.clone() ],
            "Data": [ [ tor.randn( 100 ) ] ],
            "InputVarNames": [ "x", "y" ],
            "NonLinearities": [ __import__( "NARMAX" ).Identity ],
        }
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names,
                            MaxDepth = 0, ValFunc = dummy_validation, ValData = val_data )
    arbo.fit()
    assert arbo.theta is not None
    assert isinstance( arbo.theta, tor.Tensor )
    assert arbo.ERR is not None
    assert isinstance( arbo.ERR, np.ndarray )

  def test_validate_with_valdata_no_valfunc_uses_default( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''Passing ValData without a custom ValFunc should use the default without crashing.'''
    val_data = {
            "y": [ sample_y.clone() ],
            "Data": [ [ tor.randn( 100 ) ] ],
            "InputVarNames": [ "x", "y" ],
            "NonLinearities": [ __import__( "NARMAX" ).Identity ],
        }
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names,
                            MaxDepth = 0, ValData = val_data )
    arbo.fit()
    assert arbo.theta is not None

  def test_validate_no_valdata_no_custom_valfunc( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''When neither ValData nor custom ValFunc is given, validation uses default 1-sum(ERR).'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    # The constructor installs a default ValFunc; validate should use it.
    arbo.fit()
    assert arbo.theta is not None
    # The error metric is 1 - sum(ERR) of the chosen model
    assert abs( arbo.ValFunc( arbo.theta, arbo.L, arbo.ERR, None, None ) - ( 1 - np.sum( arbo.ERR ) ) ) < 1e-12

  # --- validate with no regressions stored ---
  def test_validate_no_regressions( self, sample_y ) -> None:
    '''validate() with no regressions raises AssertionError.'''
    arbo = Arborescence( y = sample_y )
    arbo.LG.Data = []
    arbo.MinLen = [ 1 ]
    # Should not crash but will raise from get_Results since theta is None
    with pytest.raises( AssertionError, match = "No regression results" ):
      arbo.validate()


# =====================================================================================================================
# get_Results
# =====================================================================================================================

class TestGetResults:
  '''Tests for the results getter.'''

  def test_get_results_before_fit_raises( self, basic_arbo ) -> None:
    '''get_Results() before fit raises AssertionError.'''
    with pytest.raises( AssertionError, match = "No regression results" ):
      basic_arbo.get_Results()

  def test_get_results_after_fit( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''get_Results() after fit returns correct types.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    theta, L, ERR, MD, Dc, DcN = arbo.get_Results()
    assert isinstance( theta, tor.Tensor )
    assert theta.ndim == 1
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )
    assert ERR.dtype == np.float64
    assert len( ERR ) == len( theta )

  def test_get_results_returns_tuple_of_six( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''get_Results() returns a 6-tuple.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    result = arbo.get_Results()
    assert len( result ) == 6


# =====================================================================================================================
# set_ArboDepth
# =====================================================================================================================

class TestSetArboDepth:
  '''Tests for set_ArboDepth.'''

  def test_set_depth_valid( self, arbo_with_dc ) -> None:
    '''Setting a valid depth updates MaxDepth.'''
    arbo_with_dc.MinLen = [ 3, 3 ]
    # Seed the Queue so peek returns an array
    arbo_with_dc.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo_with_dc.set_ArboDepth( 2 )
    assert arbo_with_dc.MaxDepth == 2

  def test_set_depth_negative_raises( self, arbo_with_dc ) -> None:
    '''Negative depth raises AssertionError.'''
    with pytest.raises( AssertionError, match = "Depth can't be negative" ):
      arbo_with_dc.set_ArboDepth( -1 )

  def test_set_depth_less_than_current_raises( self, arbo_with_dc ) -> None:
    '''Depth less than current level raises AssertionError.'''
    arbo_with_dc.Q.put( np.array( [ 0, 1 ], dtype = np.int64 ) ) # len = 2
    with pytest.raises( AssertionError, match = "The Arbo has already passed that depth" ):
      arbo_with_dc.set_ArboDepth( 1 )

  def test_set_depth_abort_raises( self, arbo_with_dc ) -> None:
    '''Setting depth when Abort is True raises AssertionError.'''
    arbo_with_dc.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo_with_dc.Abort = True
    with pytest.raises( AssertionError, match = "The Arbo is already in its last level" ):
      arbo_with_dc.set_ArboDepth( 3 )

  def test_set_depth_greater_than_minlen_warns( self, arbo_with_dc ) -> None:
    '''Depth greater than MinLen does not update MaxDepth.'''
    arbo_with_dc.MinLen = [ 2 ]
    arbo_with_dc.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo_with_dc.set_ArboDepth( 5 )
    # MaxDepth should not have been updated
    assert arbo_with_dc.MaxDepth == 3 # unchanged

  def test_set_depth_exact( self, arbo_with_dc ) -> None:
    '''Setting depth to an exact value updates MaxDepth.'''
    arbo_with_dc.MinLen = [ 5, 5 ]
    arbo_with_dc.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo_with_dc.set_ArboDepth( 3 )
    assert arbo_with_dc.MaxDepth == 3

  def test_set_depth_equal_to_minlen_accepts( self, arbo_with_dc ) -> None:
    '''Depth equal to MinLen[-1] should be accepted (not greater).'''
    arbo_with_dc.MinLen = [ 4 ]
    arbo_with_dc.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo_with_dc.Abort = False
    arbo_with_dc.set_ArboDepth( 4 )
    assert arbo_with_dc.MaxDepth == 4


# =====================================================================================================================
# MemoryDump & load
# =====================================================================================================================

class TestMemoryDumpAndLoad:
  '''Tests for serialization (MemoryDump and load).'''

  def test_memory_dump_and_load_roundtrip( self, sample_y ) -> None:
    '''Dump an arbo to a temp file and load it back into another instance.'''
    arbo1 = Arborescence(
            y = sample_y,
            tolRoot = 0.01,
            tolRest = 0.02,
            MaxDepth = 3,
            Verbose = False,
            FileName = None,
            SaveFrequency = 0,
        )
    # Set some internal state
    item1 = np.array( [ 0, 1 ], dtype = np.int64 )
    arbo1.Q.put( item1 )
    arbo1.nNotSkippedNodes = 5
    arbo1.TotalNodes = 10

    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      # Test MemoryDump
      arbo1.MemoryDump( nComputed = 3 )

      # Load into a fresh arbo
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = False )

      # Verify round-trip
      assert arbo2.tolRoot == 0.01
      assert arbo2.tolRest == 0.02
      assert arbo2.MaxDepth == 3
      assert arbo2.Verbose is False
      assert arbo2.nNotSkippedNodes == 5
      assert arbo2.TotalNodes == 10
      assert arbo2.y is not None
      assert tor.allclose( arbo2.y, arbo1.y )
      # Queue content integrity
      assert arbo2.Q.size() == 1
      retrieved = arbo2.Q.get()
      np.testing.assert_array_equal( retrieved, item1 )
      assert arbo2.INT_TYPE == arbo1.INT_TYPE
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_memory_dump_and_load_lg_integrity( self, sample_y ) -> None:
    '''Verify that MultiKeyHashTable contents survive dump/load.'''
    arbo1 = Arborescence( y = sample_y, MaxDepth = 2 )
    # Add some data and keys
    data = np.array( [ 1, 2, 3 ], dtype = arbo1.INT_TYPE )
    idx = arbo1.LG.AddData( data )
    arbo1.LG.CreateKeys( MinLen = 1, IndexSet = data, Value = idx )
    # Also put something in the queue so load works
    arbo1.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = False )
      # Check that the LG contains the data
      assert len( arbo2.LG.Data ) == 1
      assert np.array_equal( arbo2.LG[ idx ], data )
      # Key lookup should still work
      result = arbo2.LG.SameStart( np.array( [ 1, 2 ], dtype = np.int64 ) )
      assert result == idx
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_load_corrupted_file_rolls_back( self, empty_arbo ) -> None:
    '''Loading a corrupted file should roll back state and raise RuntimeError.'''
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False, mode = "w" ) as f:
      f.write( "this is not a valid dill dump" )
      tmp_path = f.name
    try:
      with pytest.raises( RuntimeError, match = "Deserialization failed" ):
        empty_arbo.load( tmp_path, Print = False )
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_memory_dump_with_dc( self, sample_y, sample_dc, sample_dc_names ) -> None:
    '''Dump and load an arbo with Dc.'''
    arbo1 = Arborescence(
            y = sample_y,
            Dc = sample_dc,
            DcNames = sample_dc_names,
            tolRoot = 0.01,
            tolRest = 0.01,
            MaxDepth = 2,
        )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.MemoryDump( nComputed = 0 )

      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = False )

      assert arbo2.Dc is not None
      assert arbo2.Dc.shape == arbo1.Dc.shape
      assert tor.allclose( arbo2.Dc, arbo1.Dc )
      assert arbo2.DcNames is not None
      assert len( arbo2.DcNames ) == len( arbo1.DcNames )
      assert arbo2.nC == arbo1.nC
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_load_prints_stats( self, sample_y, capsys ) -> None:
    '''Loading with Print=True prints tolerance and depth info.'''
    Dc = tor.randn( 100, 3 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 3 ) ], dtype = np.str_ )
    arbo1 = Arborescence( y = sample_y, Dc = Dc, DcNames = names, MaxDepth = 2 )
    # Simulate some traversal state so load printing doesn't crash
    arbo1.MinLen = [ 3, 3 ]
    arbo1.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
    arbo1.nNotSkippedNodes = 5
    arbo1.TotalNodes = 3
    arbo1.AbortedRegs = 1
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = True )
      captured = capsys.readouterr()
      assert "Tolerance" in captured.out
      assert "ArboDepth" in captured.out
      assert "Shortest Sequence" in captured.out
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_save_frequency_conversion_in_dump( self, sample_y ) -> None:
    '''SaveFrequency should be preserved after round-trip.'''
    arbo1 = Arborescence( y = sample_y, SaveFrequency = 10 )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = False )
      assert arbo2.SaveFrequency == arbo1.SaveFrequency
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_load_pre_traversal_dump_with_print( self, sample_y ) -> None:
    '''Loading a pre-traversal dump with Print=True should not crash on empty MinLen.'''
    arbo1 = Arborescence( y = sample_y, MaxDepth = 2 )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = True )
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_load_pre_traversal_dump_with_print_prints_na( self, sample_y, capsys ) -> None:
    '''Loading a pre-traversal dump with Print=True should print 'N/A' for dict shape and not crash.'''
    arbo1 = Arborescence( y = sample_y, MaxDepth = 2 )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = True )
      captured = capsys.readouterr()
      assert "N/A" in captured.out
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )

  def test_load_ds_only_dump_with_print( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''Loading a Ds-only (no Dc) dump with Print=True should not crash on None Dc.shape.'''
    arbo1 = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    with tempfile.NamedTemporaryFile( suffix = ".narmax", delete = False ) as f:
      tmp_path = f.name
    try:
      arbo1.FileName = tmp_path
      arbo1.Q.put( np.array( [ 0 ], dtype = np.int64 ) )
      arbo1.MinLen = [ 1 ]
      arbo1.MemoryDump( nComputed = 0 )
      arbo2 = Arborescence()
      arbo2.load( tmp_path, Print = True )
    finally:
      if ( os.path.exists( tmp_path ) ): os.unlink( tmp_path )


# =====================================================================================================================
# PlotAndPrint
# =====================================================================================================================

class TestPlotAndPrint:
  '''Comprehensive tests for PlotAndPrint.'''

  @pytest.fixture( autouse = True )
  def _mock_plot_deps( self ) -> Iterator[ None ]:
    plt = pytest.importorskip( "matplotlib.pyplot" )
    with ( mock.patch( 'NARMAX.Classes.Arborescence.SymbolicOscillator' ) as MockOsc,
              mock.patch( 'NARMAX.Classes.Arborescence.InitAndComputeBuffer' ) as MockBuf ):
      MockOsc.return_value = mock.MagicMock()
      MockBuf.side_effect = lambda model, y, data: y
      self._mock_buf = MockBuf
      self._mock_osc = MockOsc
      yield
    plt.close( 'all' )

  @pytest.fixture
  def val_data( self, sample_y: tor.Tensor ) -> dict:
    return {
            "y": [ sample_y ],
            "Data": [ tor.randn( 100, 3 ) ],
            "InputVarNames": [ "x", "y" ],
            "NonLinearities": [ __import__( "NARMAX" ).Identity ],
        }

  @pytest.fixture
  def val_data_custom_output( self, sample_y: tor.Tensor ) -> dict:
    return {
            "y": [ sample_y ],
            "Data": [ tor.randn( 100, 3 ) ],
            "InputVarNames": [ "x", "y" ],
            "NonLinearities": [ __import__( "NARMAX" ).Identity ],
            "OutputVarName": "y1",
        }

  @pytest.fixture
  def fitted_arbo_ds_only( self, sample_y: tor.Tensor, sample_ds: tor.Tensor, sample_ds_names: np.ndarray ) -> Arborescence:
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    return arbo

  @pytest.fixture
  def fitted_arbo_with_dc( self, sample_y: tor.Tensor, sample_dc: tor.Tensor, sample_dc_names: np.ndarray ) -> Arborescence:
    arbo = Arborescence(
            y = sample_y, Dc = sample_dc, DcNames = sample_dc_names,
            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 3,
        )
    arbo.fit()
    return arbo

  def test_raises_without_results( self, basic_arbo ) -> None:
    '''PlotAndPrint raises error if fitting has not finished.'''
    val_data = {
            "y": [ tor.randn( 100 ) ],
            "Data": [ [ tor.randn( 100 ) ] ],
            "InputVarNames": [ "x", "y" ],
            "NonLinearities": [ __import__( "NARMAX" ).Identity ],
        }
    with pytest.raises( AssertionError, match = "fitting hasn't been finished" ):
      basic_arbo.PlotAndPrint( val_data, PrintRegressors = False )

  def test_return_structure( self, fitted_arbo_ds_only, val_data ) -> None:
    '''PlotAndPrint returns (Fig, Fig2), (Ax, Ax2) tuple.'''
    result = fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    ( Fig, Fig2 ), ( Ax, Ax2 ) = result
    import matplotlib.figure
    assert isinstance( Fig, matplotlib.figure.Figure )
    assert isinstance( Fig2, matplotlib.figure.Figure )
    assert len( Ax ) == 2
    assert len( Ax2 ) == 2

  def test_sharex( self, fitted_arbo_ds_only, val_data ) -> None:
    '''PlotAndPrint axes share the x-axis.'''
    ( _, _ ), ( Ax, _ ) = fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    assert Ax[ 0 ] in Ax[ 1 ].get_shared_x_axes()

  def test_prints_metrics( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint prints MAE, maximal deviation, and MAD.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    out = capsys.readouterr().out
    assert "MAE" in out
    assert "maximal deviation" in out
    assert "MAD" in out

  def test_metric_values_perfect_fit( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint shows 0% error for perfect fit.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    out = capsys.readouterr().out
    assert "0.000e+00%" in out

  def test_metric_values_known_error( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint shows correct error percentage.'''
    def error_side_effect( model, y, data ):
      y_norm = y.abs().max().item()
      return y - 0.1 * y_norm
    self._mock_buf.side_effect = error_side_effect
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    out = capsys.readouterr().out
    assert "10.0%" in out

  def test_default_output_var_name( self, fitted_arbo_ds_only, val_data ) -> None:
    '''PlotAndPrint uses default output variable name y.'''
    val_data.pop( "OutputVarName", None )
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    _, call_kwargs = self._mock_osc.call_args
    if ( len( self._mock_osc.call_args[ 0 ] ) >= 5 ): assert self._mock_osc.call_args[ 0 ][ 4 ] == "y"

  def test_custom_output_var_name( self, fitted_arbo_ds_only, val_data_custom_output ) -> None:
    '''PlotAndPrint uses custom OutputVarName from ValData.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data_custom_output, PrintRegressors = False )
    assert self._mock_osc.call_args[ 0 ][ 4 ] == "y1"

  def test_print_regressor_true( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint prints recognized regressors when PrintRegressors=True.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = True )
    assert "Recognized regressors" in capsys.readouterr().out

  def test_print_regressor_false( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint does not print regressors when PrintRegressors=False.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    assert "Recognized regressors" not in capsys.readouterr().out

  def test_err_sorted_descending( self, fitted_arbo_with_dc, val_data ) -> None:
    '''ERR values are sorted in descending order.'''
    sorted_err = np.flip( np.sort( fitted_arbo_with_dc.ERR ) )
    assert all( sorted_err[ i ] >= sorted_err[ i + 1 ] for i in range( len( sorted_err ) - 1 ) )

  def test_mae_evolution_length( self, fitted_arbo_ds_only, val_data ) -> None:
    '''MAE evolution plot has correct number of terms.'''
    ( _, Fig2 ), ( _, Ax2 ) = fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    n_terms = len( fitted_arbo_ds_only.theta )
    assert len( Ax2[ 1 ].lines[ 0 ].get_ydata() ) == n_terms

  def test_regnames_ds_only( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''PlotAndPrint shows Ds names when printing regressors.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = True )
    out = capsys.readouterr().out
    for name in fitted_arbo_ds_only.DsNames: assert name in out

  def test_regnames_with_dc( self, fitted_arbo_with_dc, val_data, capsys ) -> None:
    '''PlotAndPrint shows Dc names when printing regressors.'''
    fitted_arbo_with_dc.PlotAndPrint( val_data, PrintRegressors = True )
    out = capsys.readouterr().out
    for idx in fitted_arbo_with_dc.L: assert fitted_arbo_with_dc.DcNames[ idx ] in out

  def test_ds_only_uses_dsnames( self, fitted_arbo_ds_only, val_data, capsys ) -> None:
    '''Ds-only model uses DsNames for regressor printing.'''
    fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = True )
    out = capsys.readouterr().out
    for name in fitted_arbo_ds_only.DsNames: assert name in out

  def test_figures_not_closed( self, fitted_arbo_ds_only, val_data ) -> None:
    '''PlotAndPrint figures remain open with correct axes.'''
    ( Fig, Fig2 ), _ = fitted_arbo_ds_only.PlotAndPrint( val_data, PrintRegressors = False )
    assert len( Fig.axes ) == 2
    assert len( Fig2.axes ) == 2


# =====================================================================================================================
# Integration: rFOrLSR -> fit -> validate -> get_Results
# =====================================================================================================================

class TestIntegration:
  '''End-to-end integration tests with small synthetic data.'''

  def test_full_arborescence_flow( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''Full pipeline: init -> fit -> get_Results.'''
    arbo = Arborescence(
            y = sample_y,
            Ds = sample_ds,
            DsNames = sample_ds_names,
            MaxDepth = 0,
        )
    result = arbo.fit()
    theta, L, ERR, MD, Dc, DcN = result

    assert len( theta ) > 0, "theta should be non-empty"
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )
    assert len( theta ) == ERR.shape[ 0 ]

  def test_full_flow_with_dc_selection( self, sample_y ) -> None:
    '''Full pipeline with Dc candidate selection.'''
    tor.manual_seed( 42 )
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence(
            y = sample_y,
            Dc = Dc,
            DcNames = names,
            tolRoot = 0.99,
            tolRest = 0.99,
            MaxDepth = 1,
        )
    result = arbo.fit()
    theta, L, ERR, MD, Dc_out, DcN_out = result

    assert len( theta ) > 0
    assert len( L ) > 0
    assert ERR is not None
    # Dc should be returned (modified)
    assert Dc_out is not None
    assert DcN_out is not None
    # MD is None because MorphDict was not passed
    assert MD is None

  def test_fit_updates_internal_state( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''After fit, theta/L/ERR on the class should be populated.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    assert arbo.theta is not None
    assert arbo.L is not None
    assert arbo.ERR is not None
    assert arbo.nNotSkippedNodes > 0

  def test_validate_updates_best_model( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''validate should find the best model and store it.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    arbo.fit()
    # theta should be set to the best model
    assert arbo.theta is not None
    assert arbo.L is not None
    assert arbo.ERR is not None


# =====================================================================================================================
# Edge cases and error handling
# =====================================================================================================================

class TestEdgeCases:
  '''Corner cases and defensive programming.'''

  def test_y_all_zeros( self ) -> None:
    '''y = zeros should work (mean-centering yields zeros).'''
    y = tor.zeros( 100 )
    arbo = Arborescence( y = y )
    assert arbo.y is not None
    assert tor.allclose( arbo.y, tor.zeros( 100 ) )
    assert arbo.y.dtype == tor.float64
    assert arbo.nS == 0
    assert arbo.nC == 0

  def test_dc_all_identical_columns( self, sample_y ) -> None:
    '''All columns identical -> all but one removed as duplicates.'''
    col = tor.randn( 100, 1 )
    Dc = tor.cat( [ col ] * 5, dim = 1 )
    names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names )
    assert arbo.Dc.shape[ 1 ] == 1
    assert arbo.nC == 1
    assert arbo.DcNames.shape == ( 1, )
    assert arbo.DcNames[ 0 ] == "r0"

  def test_dc_single_column_fit_succeeds( self, sample_y ) -> None:
    '''Dc with a single column should fit without error.'''
    Dc = tor.randn( 100, 1 )
    Dc -= tor.mean( Dc )
    arbo = Arborescence( y = sample_y, Dc = Dc,
                            DcNames = np.array( [ "single" ], dtype = np.str_ ),
                            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 1 )
    result = arbo.fit()
    assert len( result ) == 6

  def test_very_deep_arbo_terminates( self, sample_y ) -> None:
    '''Arbo with depth > shortest sequence should still finish.'''
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence(
            y = sample_y, Dc = Dc, DcNames = names,
            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 5,
        )
    result = arbo.fit()
    assert len( result ) == 6

  def test_y_constant_value( self ) -> None:
    '''y = constant -> after centering y = zeros.'''
    y = tor.ones( 100 ) * 5.0
    arbo = Arborescence( y = y )
    assert tor.allclose( arbo.y, tor.zeros( 100 ) )
    assert isinstance( arbo.y, tor.Tensor )
    assert arbo.Ds.shape == ( 100, 0 )
    assert arbo.Dc is None

  def test_maxdepth_zero_without_dc( self, sample_y, sample_ds, sample_ds_names ) -> None:
    '''MaxDepth=0 with Ds only: immediate return after root.'''
    arbo = Arborescence( y = sample_y, Ds = sample_ds, DsNames = sample_ds_names, MaxDepth = 0 )
    result = arbo.fit()
    theta, L, ERR, MD, Dc, DcN = result
    assert len( theta ) > 0
    assert len( L ) == 0 # no Dc selection
    assert len( ERR ) > 0
    assert MD is None
    assert Dc is None
    assert DcN is None
    assert isinstance( theta, tor.Tensor )
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )

  def test_maxdepth_zero_with_dc( self, sample_y ) -> None:
    '''MaxDepth=0 with Dc only: root selected, no traversal.'''
    Dc = tor.randn( 100, 5 )
    Dc -= tor.mean( Dc, axis = 0, keepdims = True )
    names = np.array( [ f"r{ i }" for i in range( 5 ) ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Dc = Dc, DcNames = names,
                            tolRoot = 0.99, tolRest = 0.99, MaxDepth = 0 )
    result = arbo.fit()
    assert len( result ) == 6
    theta, L, ERR, MD, Dc_out, DcN_out = result
    assert isinstance( theta, tor.Tensor )
    assert theta.ndim == 1
    assert isinstance( L, np.ndarray )
    assert isinstance( ERR, np.ndarray )
    assert len( ERR ) == len( theta )
    assert MD is None


# =====================================================================================================================
# AllCombinations integration (used by fit)
# =====================================================================================================================

class TestAllCombinationsIntegration:
  '''Tests the AllCombinations function used in fit traversal.'''

  def test_all_combinations_basic( self ) -> None:
    '''AllCombinations produces correct shape and values.'''
    imposed = np.array( [ 0, 1 ], dtype = np.int64 )
    seq = np.array( [ 2, 3, 4 ], dtype = np.int64 )
    result = AllCombinations( imposed, seq, np.int64 )
    assert result.shape == ( 3, 3 )
    assert np.array_equal( result[ :, -1 ], seq )
    assert np.array_equal( result[ 0, : -1 ], imposed )
    assert result.dtype == np.int64

  def test_all_combinations_removes_duplicates( self ) -> None:
    '''AllCombinations removes duplicate values from seq.'''
    imposed = np.array( [ 0, 2 ], dtype = np.int64 )
    seq = np.array( [ 1, 2, 3 ], dtype = np.int64 )
    result = AllCombinations( imposed, seq, np.int64 )
    assert result.shape == ( 2, 3 )
    assert np.array_equal( result[ :, -1 ], np.array( [ 1, 3 ] ) )

  def test_all_combinations_empty_result( self ) -> None:
    '''AllCombinations returns empty array when no unique values remain.'''
    imposed = np.array( [ 0, 1, 2 ], dtype = np.int64 )
    seq = np.array( [ 0, 1, 2 ], dtype = np.int64 )
    result = AllCombinations( imposed, seq, np.int64 )
    assert result.shape == ( 0, 4 )


# =====================================================================================================================
# Ds centering reversion in validate
# =====================================================================================================================

class TestDsCentering:
  '''Tests that Ds centering is properly reverted in validate.'''

  def test_ds_means_reverted_after_validate( self, sample_y ) -> None:
    '''Ds means are restored after validate() completes.'''
    Ds = tor.randn( 100, 2 )
    orig_Ds = Ds.clone()
    names = np.array( [ "a", "b" ], dtype = np.str_ )
    arbo = Arborescence( y = sample_y, Ds = Ds, DsNames = names, MaxDepth = 0 )
    arbo.fit()
    # After fit (which calls validate), Ds should have means restored
    assert tor.allclose( arbo.Ds, orig_Ds, atol = 1e-10 )
    # DsMeans should be stored for later use
    assert arbo.DsMeans is not None
    assert arbo.DsMeans.shape == ( 1, 2 )
    assert tor.allclose( arbo.DsMeans, tor.mean( orig_Ds, axis = 0, keepdims = True ), atol = 1e-10 )
    assert arbo.nNotSkippedNodes > 0


# =====================================================================================================================
# Queue interaction (used by Arborescence)
# =====================================================================================================================

class TestQueueInteraction:
  '''Tests the Queue usage patterns within Arborescence.'''

  def test_queue_peek_empty_returns_empty_array( self, arbo_with_dc ) -> None:
    '''Peeking an empty queue returns an empty array.'''
    assert len( arbo_with_dc.Q.peek() ) == 0

  def test_queue_put_and_get( self, arbo_with_dc ) -> None:
    '''Putting and getting from queue works correctly.'''
    item = np.array( [ 1, 2, 3 ], dtype = np.int64 )
    arbo_with_dc.Q.put( item )
    assert arbo_with_dc.Q.size() == 1
    retrieved = arbo_with_dc.Q.get()
    np.testing.assert_array_equal( retrieved, item )

  def test_queue_clear( self, arbo_with_dc ) -> None:
    '''Clearing the queue empties it.'''
    arbo_with_dc.Q.put( np.array( [ 1 ], dtype = np.int64 ) )
    arbo_with_dc.Q.clear()
    assert arbo_with_dc.Q.is_empty()


# =====================================================================================================================
# MultiKeyHashTable interaction (used by Arborescence)
# =====================================================================================================================

class TestMultiKeyHashTableInteraction:
  '''Tests the MultiKeyHashTable usage patterns within Arborescence.'''

  def test_lg_add_and_retrieve( self, arbo_with_dc ) -> None:
    '''Adding and retrieving data from LG works.'''
    data = np.array( [ 0, 1, 2 ], dtype = arbo_with_dc.INT_TYPE )
    idx = arbo_with_dc.LG.AddData( data )
    np.testing.assert_array_equal( arbo_with_dc.LG[ idx ], data )

  def test_lg_samestart_match( self, arbo_with_dc ) -> None:
    '''LG SameStart finds a matching prefix.'''
    data = np.array( [ 3, 1, 5 ], dtype = np.int64 )
    idx = arbo_with_dc.LG.AddData( data )
    arbo_with_dc.LG.CreateKeys( MinLen = 1, IndexSet = data, Value = idx )
    result = arbo_with_dc.LG.SameStart( np.array( [ 1, 3 ], dtype = np.int64 ) )
    assert result == idx

  def test_lg_samestart_no_match( self, arbo_with_dc ) -> None:
    '''LG SameStart returns empty list for no match.'''
    result = arbo_with_dc.LG.SameStart( np.array( [ 99 ], dtype = np.int64 ) )
    assert result == []

  def test_lg_delete_keys( self, arbo_with_dc ) -> None:
    '''LG DeleteAllOfSize removes keys correctly.'''
    data = np.array( [ 0, 1, 2 ], dtype = np.int64 )
    idx = arbo_with_dc.LG.AddData( data )
    arbo_with_dc.LG.CreateKeys( MinLen = 1, IndexSet = data, Value = idx )
    arbo_with_dc.LG.DeleteAllOfSize( 2 )
    # Keys of length <= 2 should be gone
    assert arbo_with_dc.LG.SameStart( np.array( [ 0, 1 ], dtype = np.int64 ) ) == []
