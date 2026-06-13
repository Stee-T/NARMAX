import pytest
import torch as tor
import numpy as np
from unittest.mock import patch, MagicMock
from numpy import int64, float64
from numpy.typing import NDArray
from typing import Any, Optional

from NARMAX.Classes.NonLinearity import NonLinearity
from NARMAX.Validation import DefaultValidation

# ----------------------------------------------------------------------
# Common fixtures

@pytest.fixture
def dummy_theta() -> tor.Tensor:
  '''A valid theta tensor (arbitrary size).'''
  return tor.tensor( [ 1.0, -0.5, 0.2 ], dtype = tor.float32 )


@pytest.fixture
def dummy_L() -> NDArray[ int64 ]: return np.array( [ 1, 2, 3 ], dtype = int64 )


@pytest.fixture
def dummy_ERR() -> NDArray[ float64 ]: return np.array( [ 0.9, 0.05, 0.01 ], dtype = float64 )


@pytest.fixture
def dummy_reg_names() -> NDArray[ np.str_ ]: return np.array( [ "u(t-1)", "y(t-1)", "u(t-2)" ], dtype = np.str_ )


@pytest.fixture
def dummy_input_var_names() -> list[ str ]: return [ "u", "y" ]


@pytest.fixture
def dummy_nonlin() -> list[ NonLinearity ]:
  '''A list of NonLinearity instances.'''
  return [ NonLinearity( "id", lambda x : x ), NonLinearity( "abs", tor.abs ) ]


@pytest.fixture
def valid_y_single() -> tor.Tensor:
  '''A single validation output tensor.'''
  return tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ], dtype = tor.float32 )


@pytest.fixture
def valid_data_single( valid_y_single: tor.Tensor ) -> list[ tor.Tensor ]:
  '''A matching Data entry (list of tensors).'''
  return [ tor.tensor( [ 10.0, 20.0, 30.0, 40.0 ], dtype = tor.float32 ) ]


@pytest.fixture
def valid_valdata_single( dummy_input_var_names: list[ str ], dummy_nonlin: list[ NonLinearity ], valid_y_single: tor.Tensor, valid_data_single: list[ tor.Tensor ] ) -> dict[ str, Any ]:
  '''A fully valid ValData dict with one validation set.'''
  return {
        "y": [ valid_y_single ],
        "Data": [ valid_data_single ],
        "InputVarNames": dummy_input_var_names,
        "NonLinearities": dummy_nonlin,
    }


@pytest.fixture
def valid_valdata_multi( valid_valdata_single: dict[ str, Any ] ) -> dict[ str, Any ]:
  '''A valid ValData dict with two validation sets (duplicate same data).'''
  vd = valid_valdata_single.copy()
  vd[ "y" ] = [ vd[ "y" ][ 0 ], vd[ "y" ][ 0 ] * 2 ]
  vd[ "Data" ] = [ vd[ "Data" ][ 0 ], vd[ "Data" ][ 0 ] ]
  return vd


@pytest.fixture
def mock_symbolic_oscillator() -> MagicMock:
  '''Patch SymbolicOscillator to return a MagicMock.'''
  with patch( "NARMAX.Validation.SymbolicOscillator" ) as mock:
    yield mock


@pytest.fixture
def mock_init_and_compute() -> MagicMock:
  '''Patch InitAndComputeBuffer to return a controllable yHat.'''
  with patch( "NARMAX.Validation.InitAndComputeBuffer" ) as mock:
    yield mock

# ----------------------------------------------------------------------
# Tests for input validation (AssertionError cases)

class TestInputValidation:
  def test_valdata_not_dict( self, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''ValData not a dict raises AssertionError.'''
    with pytest.raises( AssertionError, match = "datastructure is not a dictionary" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, "not_a_dict" )

  @pytest.mark.parametrize( "missing_key", [ "y", "InputVarNames", "Data", "NonLinearities" ] )
  def test_missing_required_key( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, missing_key ) -> None:
    '''Missing required keys raise AssertionError.'''
    vd = valid_valdata_single.copy()
    del vd[ missing_key ]
    with pytest.raises( AssertionError, match = f"contains no '{ missing_key }' entry" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_y_not_list( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''y not being a list raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "y" ] = "not_a_list"
    with pytest.raises( AssertionError, match = "expected to be a list" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_is_none( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Data being None raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = None
    with pytest.raises( AssertionError, match = "'Data' entry is None" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_not_list( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Data not being a list raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = np.array( [ 1, 2 ] ) # numpy array passes first None check but not list
    with pytest.raises( AssertionError, match = "expected to be a list" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_empty_list( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Empty Data list raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = []
    with pytest.raises( AssertionError, match = "empty, there is thus nothing to validation against" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_y_length_mismatch( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Length mismatch between y and Data raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "y" ] = [ vd[ "y" ][ 0 ], vd[ "y" ][ 0 ] ] # 2 y's
    vd[ "Data" ] = [ vd[ "Data" ][ 0 ] ] # only 1 Data
    with pytest.raises( AssertionError, match = "same length" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_y_element_not_tensor( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''y element not a tensor raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "y" ] = [ 123.0 ] # float, not tensor
    with pytest.raises( AssertionError, match = "not a torch.Tensor" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_y_element_scalar( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Scalar tensor in y raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "y" ] = [ tor.tensor( 5.0 ) ] # scalar tensor
    with pytest.raises( AssertionError, match = "scalar tensor" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_sublist_not_list_or_tuple( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Data sublist not list/tuple raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = [ "not_a_list" ]
    with pytest.raises( AssertionError, match = "not a list/tuple" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_sublist_empty( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Empty Data sublist raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = [ [] ] # empty sub-list
    with pytest.raises( AssertionError, match = "empty; the model expects at least one input tensor" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_data_sublist_contains_non_tensor( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Non-tensor in Data sublist raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = [ [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] ), "not_tensor" ] ]
    with pytest.raises( AssertionError, match = "expected to be a list of 2D-tuples of float tor.Tensors" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_shape_mismatch_y_vs_reg( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''Shape mismatch between y and Data raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "y" ] = [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] # length 3
    vd[ "Data" ] = [ [ tor.tensor( [ 10.0, 20.0, 30.0, 40.0 ] ) ] ] # length 4
    with pytest.raises( AssertionError, match = "not the same length" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_nonlinearities_not_list( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''NonLinearities not a list raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "NonLinearities" ] = NonLinearity( "id", lambda x: x ) # single object, not list
    with pytest.raises( AssertionError, match = "expected to be a list" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_nonlinearities_item_not_nonlinearity( self, valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''NonLinearities item not NonLinearity raises AssertionError.'''
    vd = valid_valdata_single.copy()
    vd[ "NonLinearities" ] = [ NonLinearity( "id", lambda x : x ), "not_a_nonlin" ]
    with pytest.raises( AssertionError, match = "expected to be a list of NonLinearity objects" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )

  def test_valdata_is_none( self, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names ) -> None:
    '''ValData being None raises AssertionError.'''
    with pytest.raises( AssertionError, match = "datastructure is not a dictionary" ):
      DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, None )

# ----------------------------------------------------------------------
# Tests for correct behaviour (with mocked dependencies)

class TestCorrectBehaviour:
  def test_creates_symbolic_oscillator_with_correct_args(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Verify that SymbolicOscillator is instantiated with the right arguments.'''
    mock_init_and_compute.return_value = tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] )
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_single )
    mock_symbolic_oscillator.assert_called_once_with(
            valid_valdata_single[ "InputVarNames" ],
            valid_valdata_single[ "NonLinearities" ],
            dummy_reg_names,
            dummy_theta,
            "y" # default OutputVarName
        )
    assert isinstance( result, float )
    assert result == 0.0  # perfect prediction → error = 0

  def test_uses_output_var_name_when_provided(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''If OutputVarName is in ValData, it must be passed to the model.'''
    vd = valid_valdata_single.copy()
    vd[ "OutputVarName" ] = "z"
    mock_init_and_compute.return_value = tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] )
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )
    mock_symbolic_oscillator.assert_called_once_with(
            vd[ "InputVarNames" ],
            vd[ "NonLinearities" ],
            dummy_reg_names,
            dummy_theta,
            "z"
        )
    assert isinstance( result, float )
    assert result == 0.0  # perfect prediction → error = 0

  def test_dc_filter_idx_accepted_but_unused(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Passing DcFilterIdx should not interfere with normal operation.'''
    mock_init_and_compute.return_value = tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] )
    dc = np.array( [ 1, 2 ], dtype = int64 )
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names,
                                   valid_valdata_single, DcFilterIdx = dc )
    assert isinstance( result, float )
    assert result >= 0.0

  def test_error_computation_single_set(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Check that the returned error is the relative MAE for one validation set.'''
    y = valid_valdata_single[ "y" ][ 0 ] # tensor([1.,2.,3.,4.])
    # Make InitAndComputeBuffer return an array that yields a known error
    # Let yHat be y + 0.1 (absolute error 0.1 each)
    yhat = y + 0.1
    mock_init_and_compute.return_value = yhat
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_single )
    # Expected error = mean(abs(y - yhat) / mean(abs(y)))
    expected = tor.mean( tor.abs( y - yhat ) / tor.mean( tor.abs( y ) ) ).item()
    assert isinstance( result, float )
    assert result >= 0.0
    assert result == pytest.approx( expected )
    # Ensure InitAndComputeBuffer was called with correct arguments
    mock_init_and_compute.assert_called_once_with(
            mock_symbolic_oscillator.return_value,
            y,
            valid_valdata_single[ "Data" ][ 0 ]
        )

  def test_error_computation_multiple_sets(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_multi, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Average of relative errors over multiple validation sets.'''
    y1 = valid_valdata_multi[ "y" ][ 0 ] # [1,2,3,4]
    y2 = valid_valdata_multi[ "y" ][ 1 ] # [2,4,6,8]
    # Simulate that InitAndComputeBuffer returns identical predictions
    # for simplicity, yHat = y (perfect prediction) -> error 0
    mock_init_and_compute.side_effect = [ y1, y2 ]
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_multi )
    assert isinstance( result, float )
    assert result == 0.0  # exact zero for perfect prediction
    # Verify two calls
    assert mock_init_and_compute.call_count == 2

  def test_zero_target_handling(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''When mean(abs(y)) == 0, denom is set to 1e-8 to avoid division by zero.'''
    vd = valid_valdata_single.copy()
    y = tor.zeros( 4 )
    vd[ "y" ] = [ y ]
    # Make prediction also zero → abs difference = 0
    mock_init_and_compute.return_value = tor.zeros( 4 )
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )
    # Error per sample: mean(|0-0|) / 1e-8 = 0.0 / 1e-8 = 0.0
    assert isinstance( result, float )
    assert result == 0.0

    # If prediction differs, error should be computed with epsilon denominator
    yhat = tor.ones( 4 ) * 2.0
    mock_init_and_compute.return_value = yhat
    result2 = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )
    assert isinstance( result2, float )
    assert result2 >= 0.0
    expected = ( tor.mean( tor.abs( y - yhat ) ) / 1e-8 ).item()
    assert result2 == pytest.approx( expected )

  def test_the_no_op_self_assignment_does_not_break(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''The line ValData["Data"][val] = ValData["Data"][val] is harmless.'''
    mock_init_and_compute.return_value = valid_valdata_single[ "y" ][ 0 ]
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_single )
    assert isinstance( result, float )
    assert result == 0.0  # perfect prediction

  def test_error_computation_multiple_sets_nonzero(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_multi, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Non-zero error averaged over multiple validation sets.'''
    y1 = valid_valdata_multi[ "y" ][ 0 ] # [1,2,3,4]
    y2 = valid_valdata_multi[ "y" ][ 1 ] # [2,4,6,8]
    # Offset predictions by +0.5
    mock_init_and_compute.side_effect = [ y1 + 0.5, y2 + 0.5 ]
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_multi )
    e1 = tor.mean( tor.abs( y1 - ( y1 + 0.5 ) ) / tor.mean( tor.abs( y1 ) ) ).item()
    e2 = tor.mean( tor.abs( y2 - ( y2 + 0.5 ) ) / tor.mean( tor.abs( y2 ) ) ).item()
    assert isinstance( result, float )
    assert result >= 0.0
    assert result > 0.0  # non-zero offset → strictly positive error
    assert result == pytest.approx( ( e1 + e2 ) / 2.0 )
    assert mock_init_and_compute.call_count == 2

  def test_dc_filter_idx_default_none(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Omitting DcFilterIdx (default None) works.'''
    mock_init_and_compute.return_value = valid_valdata_single[ "y" ][ 0 ]
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, valid_valdata_single )
    assert isinstance( result, float )
    assert result == 0.0  # perfect prediction

  def test_data_sublist_as_tuple_accepted(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Data sublist can be a tuple (the validation accepts list or tuple).'''
    vd = valid_valdata_single.copy()
    vd[ "Data" ] = [ tuple( vd[ "Data" ][ 0 ] ) ]
    mock_init_and_compute.return_value = vd[ "y" ][ 0 ]
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )
    assert isinstance( result, float )
    assert result == 0.0

  def test_miso_multiple_input_regressors(
        self, mock_symbolic_oscillator, mock_init_and_compute,
        valid_valdata_single, dummy_theta, dummy_L, dummy_ERR, dummy_reg_names
    ) -> None:
    '''Multiple regressors in Data[val] (MISO) work correctly.'''
    vd = valid_valdata_single.copy()
    y = vd[ "y" ][ 0 ]  # [1,2,3,4]
    reg1 = tor.tensor( [ 10.0, 20.0, 30.0, 40.0 ] )
    reg2 = tor.tensor( [ 100.0, 200.0, 300.0, 400.0 ] )
    vd[ "Data" ] = [ [ reg1, reg2 ] ]
    mock_init_and_compute.return_value = y  # perfect prediction
    result = DefaultValidation( dummy_theta, dummy_L, dummy_ERR, dummy_reg_names, vd )
    assert isinstance( result, float )
    assert result == 0.0
    # Verify InitAndComputeBuffer got both regressors
    mock_init_and_compute.assert_called_once_with(
        mock_symbolic_oscillator.return_value, y, [ reg1, reg2 ]
    )
