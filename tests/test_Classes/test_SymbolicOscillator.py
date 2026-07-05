import pytest
import torch as tor
from typing import Optional
from NARMAX.Classes.Parser_0_3 import ExpressionParser, ExprNode
from NARMAX.Classes.SymbolicOscillator_0_4 import SymbolicOscillator, ParsedReg2EvalStr
from NARMAX.Classes.NonLinearity import NonLinearity

def make_identity_nonlin() -> NonLinearity: return NonLinearity( "id", lambda x: x )
def close( a: tor.Tensor, b: tor.Tensor, rtol: float = 1e-5 ) -> bool: return tor.allclose( a, b, rtol = rtol )


@pytest.fixture
def basic_inputs() -> list[ NonLinearity ]:
  # simple non-linearities
  return [ NonLinearity( "id", lambda x : x ), NonLinearity( "abs", tor.abs ) ]


class TestConstructor:
  def test_valid_construction( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Constructing a valid SymbolicOscillator works correctly.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x1", "x2" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x1", "x2[k-1]", "id(x1[k])" ],
            theta = tor.ones( 3 )
        )
    assert osc.get_nRegressors() == 3
    assert osc.get_nInputVars() == 2
    assert tor.allclose( osc.get_theta(), tor.ones( 3 ) )
    assert osc.get_MaxInputLag() == 1  # from "x2[k-1]"
    assert isinstance( osc.get_nRegressors(), int )
    assert isinstance( osc.get_nInputVars(), int )
    assert isinstance( osc.get_MaxInputLag(), int )
    assert isinstance( osc.get_MaxOutputLag(), int )
    assert isinstance( osc.get_MaxPositiveInputLag(), int )
    assert osc.get_MaxOutputLag() == 0
    assert osc.get_MaxPositiveInputLag() == 0

  def test_theta_length_mismatch( self, basic_inputs ) -> None:
    '''Mismatch between theta length and expression count raises ValueError.'''
    with pytest.raises( ValueError, match = "must equal number of expressions" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x1", "x1[k-1]", "id(x1)" ],
                theta = tor.ones( 2 ) # only 2, but 3 expressions
            )
    with pytest.raises( ValueError, match = "must equal number of expressions" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x1", "x1[k-1]" ],
                theta = tor.ones( 3 ) # 3, but only 2 expressions
            )

  def test_positive_output_lag_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    with pytest.raises( ValueError, match = "Positive lag for output" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1", "y" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k+1]" ],
                theta = tor.ones( 1 ),
                OutputVarName = "y"
            )

  def test_duplicate_variable_names_raises( self, basic_inputs ) -> None:
    '''Duplicate variable names in ModelVarNames raise ValueError.'''
    with pytest.raises( ValueError, match = "Duplicate names found in ModelVarNames" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1", "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x1" ],
                theta = tor.ones( 1 )
            )

  def test_output_variable_not_in_input_list( self, basic_inputs ) -> None:
    '''Using an output variable not in ModelVarNames raises ValueError.'''
    # Use output variable 'y' but not in ModelVarNames
    with pytest.raises( ValueError, match = "not declared in ModelVarNames list" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k-1]" ],
                theta = tor.ones( 1 ),
                OutputVarName = "y"
            )

  def test_positive_output_lag_raises( self, basic_inputs ) -> None:
    '''A positive lag on the output variable raises ValueError.'''
    with pytest.raises( ValueError, match = "Positive lag for output" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1", "y" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k+1]" ],
                theta = tor.ones( 1 ),
                OutputVarName = "y"
            )

  def test_missing_nonlinearity_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Using an undeclared non-linearity function name raises ValueError.'''
    with pytest.raises( ValueError, match = "not a declared Non-linearity" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "unknownFunc(x1)" ],
                theta = tor.ones( 1 )
            )

  def test_duplicate_nonlin_names_raises( self ) -> None:
    '''Duplicate non-linearity names raise ValueError.'''
    # duplicate names should be rejected
    with pytest.raises( ValueError, match = "Duplicate non-linearity names found" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = [ NonLinearity( "abs", tor.abs ), NonLinearity( "abs", tor.abs ) ],
                ExprList = [ "abs(x1)" ],
                theta = tor.ones( 1 )
            )

  def test_empty_ModelVarNames_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Empty ModelVarNames raises ValueError.'''
    with pytest.raises( ValueError, match = "No Input variables were declared" ):
      SymbolicOscillator(
                ModelVarNames = [],
                NonLinearities = basic_inputs,
                ExprList = [ "x1" ],
                theta = tor.ones( 1 )
            )

  def test_empty_NonLinearities_raises( self ) -> None:
    '''Empty NonLinearities raises ValueError.'''
    with pytest.raises( ValueError, match = "No Non-linearities were declared" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = [],
                ExprList = [ "x1" ],
                theta = tor.ones( 1 )
            )

  def test_empty_ExprList_raises( self, basic_inputs ) -> None:
    '''Empty ExprList raises ValueError.'''
    with pytest.raises( ValueError, match = "No Regressors were declared" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [],
                theta = tor.ones( 1 )
            )

  def test_theta_not_tensor_raises( self, basic_inputs ) -> None:
    '''theta not being a torch Tensor raises ValueError.'''
    with pytest.raises( ValueError, match = "theta must be of type" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x1" ],
                theta = [ 1.0, 2.0 ]
            )

  def test_reserved_name_OutVec_variable_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Using "OutVec" as a variable name raises ValueError.'''
    with pytest.raises( ValueError, match = "reserved for internal processing" ):
      SymbolicOscillator(
                ModelVarNames = [ "OutVec" ],
                NonLinearities = basic_inputs,
                ExprList = [ "OutVec[k-1]" ],
                theta = tor.ones( 1 )
            )

  def test_reserved_name_OutVec_nonlin_raises( self ) -> None:
    '''Using "OutVec" as a non-linearity name raises ValueError.'''
    with pytest.raises( ValueError, match = "reserved for internal processing" ):
      SymbolicOscillator(
                ModelVarNames = [ "x1" ],
                NonLinearities = [ NonLinearity( "OutVec", tor.abs ) ],
                ExprList = [ "OutVec(x1)" ],
                theta = tor.ones( 1 )
            )

# ###############################################################################
# 2. Oscillate functionality
# ###############################################################################

class TestInternals:
  def test_parsed_reg_to_eval_string_nested( self ) -> None:
    '''ParsedReg2EvalStr produces correct evaluation string from a ParsedReg.'''
    # AST from parser: 3.0 * sin(x1 + x2)
    node: ExprNode = ExpressionParser( "3*sin(x1 + x2)" )
    eval_str: str
    is_ar: bool
    eval_str, is_ar = ParsedReg2EvalStr( node, { "x1": 0, "x2": 1 }, { "sin": 0 }, 'y' )
    assert "NonLinList[0].get_f()" in eval_str
    assert ( "(Data[0][k])" in eval_str ) and ( "(Data[1][k])" in eval_str )
    assert not is_ar
    assert isinstance( eval_str, str )
    assert isinstance( is_ar, bool )

  def test_deep_nesting_AR_detection( self ) -> None:
    # cos( y[k-1] + x ) -> Should detect AR because of y
    node: ExprNode = ExpressionParser( "cos( y[k-1] + x )" )
    _, is_ar = ParsedReg2EvalStr( node, { "x": 0 }, { "cos": 0 }, 'y' )
    assert is_ar
    assert isinstance( is_ar, bool )

  def test_buffer_toggle( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Buffer_Toggle returns correct data using internal storage.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    osc.set_InputStorage( tor.tensor( [ [ 10.0 ] ] ) )
    assert tor.allclose(
            osc.Buffer_Toggle( 0, -1, [ tor.tensor( [ 1.0, 2.0 ] ) ] ),
            tor.tensor( [ 10.0, 1.0 ] )
        )
    assert isinstance( osc.Buffer_Toggle( 0, -1, [ tor.tensor( [ 1.0, 2.0 ] ) ] ), tor.Tensor )

  def test_parsed_reg_to_eval_string_plain( self ) -> None:
    node: ExprNode = ExpressionParser( "2.0*x1[k-1]" )
    eval_str: str
    is_ar: bool
    eval_str, is_ar = ParsedReg2EvalStr( node, { "x1": 0 }, {}, 'y' )
    assert "Data[0][k-1]" in eval_str
    assert not is_ar
    assert isinstance( eval_str, str )
    assert isinstance( is_ar, bool )

  def test_parsed_reg_to_eval_string_AR( self ) -> None:
    '''ParsedReg2EvalStr correctly identifies AR expressions.'''
    node: ExprNode = ExpressionParser( "0.5*y[k-1]" )
    _, is_ar = ParsedReg2EvalStr( node, { "x1": 0 }, {}, 'y' )
    assert is_ar
    assert isinstance( is_ar, bool )

  def test_parsed_reg_to_eval_string_with_nonlin( self ) -> None:
    '''ParsedReg2EvalStr with a non-linearity function produces correct string.'''
    node: ExprNode = ExpressionParser( "abs(x1[k])" )
    eval_str: str
    is_ar: bool
    eval_str, is_ar = ParsedReg2EvalStr( node, { "x1": 0 }, { "abs": 0 }, 'y' )
    assert "NonLinList[0].get_f()" in eval_str
    assert "(Data[0][k])" in eval_str
    assert not is_ar
    assert isinstance( eval_str, str )
    assert isinstance( is_ar, bool )

  def test_parsed_reg_to_eval_string_with_operators( self ) -> None:
    '''ParsedReg2EvalStr with multiple sub-expressions and operators.'''
    node: ExprNode = ExpressionParser( "x1[k] * x2[k-1]" )
    eval_str: str
    is_ar: bool
    eval_str, is_ar = ParsedReg2EvalStr( node, { "x1": 0, "x2": 1 }, {}, 'y' )
    assert "Data[0][k]" in eval_str
    assert "Data[1][k-1]" in eval_str
    assert "*" in eval_str
    assert not is_ar
    assert isinstance( eval_str, str )
    assert isinstance( is_ar, bool )

  def test_buffer_toggle_k0( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Buffer_Toggle with k=0 returns Data directly.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    data: list[ tor.Tensor ] = [ tor.tensor( [ 1.0, 2.0 ] ) ]
    assert tor.allclose( osc.Buffer_Toggle( 0, 0, data ), data[ 0 ] )
    assert isinstance( osc.Buffer_Toggle( 0, 0, data ), tor.Tensor )

  def test_buffer_toggle_positive_k_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Buffer_Toggle with k > 0 raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Buffer_Toggle( 0, 1, [ tor.randn( 5 ) ] )

  def test_scalar_toggle( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Scalar_Toggle returns correct scalar using storage or output vector.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]", "y[k-1]" ],
            theta = tor.tensor( [ 1.0, 1.0 ] ),
            OutputVarName = "y"
        )
    osc.set_InputStorage( tor.tensor( [ [ 5.0 ] ] ) )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    data: list[ tor.Tensor ] = [ tor.tensor( [ 1.0 ] ), tor.tensor( [ 2.0 ] ) ]
    assert osc.Scalar_Toggle( 0, 0, -1, data ) == 5.0
    osc.OutVec[ 0 ] = 3.0
    assert osc.Scalar_Toggle( 1, None, 0, data ) == 3.0
    assert osc.Scalar_Toggle( 0, 0, 0, data ) == 1.0
    assert osc.Scalar_Toggle( 1, None, -1, data ) == 7.0
    assert isinstance( osc.Scalar_Toggle( 0, 0, -1, data ), tor.Tensor )
    assert isinstance( osc.Scalar_Toggle( 1, None, 0, data ), tor.Tensor )
    assert isinstance( osc.Scalar_Toggle( 0, 0, 0, data ), tor.Tensor )
    assert isinstance( osc.Scalar_Toggle( 1, None, -1, data ), tor.Tensor )

  def test_scalar_toggle_invalid_arg_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Scalar_Toggle with invalid DataOrOutVec raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Internal Error" ):
      osc.Scalar_Toggle( 2, None, 0, [ tor.randn( 1 ) ] )

class TestOscillate:
  def test_simple_MA_system( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A simple moving-average system (no lags) produces correct output.'''
    # y[k] = theta0 * x[k]   (no lags)
    x: tor.Tensor = tor.randn( 100 )
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 2.0 ] )
        )
    result: tor.Tensor = osc.Oscillate( [ x ] )
    assert tor.allclose( result, 2.0 * x )
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert result.shape == ( 100, )
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_nested_function_execution( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Nested non-linearity function execution produces correct output.'''
    # y = id( id(x) )
    osc: SymbolicOscillator = SymbolicOscillator(
      ModelVarNames = [ "x" ],
      NonLinearities = basic_inputs,
      ExprList = [ "id(id(x))" ],
      theta = tor.tensor( [ 2.0 ] )
    )
    x: tor.Tensor = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    res: tor.Tensor = osc.Oscillate( [ x ] )
    assert tor.allclose( res, 2.0 * x )
    assert isinstance( res, tor.Tensor )
    assert str( res.device ) == osc.device
    assert res.dtype == osc.dtype
    assert res.shape == ( 3, )
    assert tor.isfinite( res ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_complex_parentheses_grouping( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Complex parenthetical grouping evaluates with correct operator precedence.'''
    # y = (x1 + x2) * x3
    osc: SymbolicOscillator = SymbolicOscillator(
      ModelVarNames = [ "x1", "x2", "x3" ],
      NonLinearities = basic_inputs,
      ExprList = [ "(x1 + x2) * x3" ],
      theta = tor.tensor( [ 1.0 ] )
    )
    x1: tor.Tensor = tor.tensor( [ 1.0, 2.0 ] )
    x2: tor.Tensor = tor.tensor( [ 3.0, 4.0 ] )
    x3: tor.Tensor = tor.tensor( [ 2.0, 2.0 ] )
    res: tor.Tensor = osc.Oscillate( [ x1, x2, x3 ] )
    assert tor.allclose( res, tor.tensor( [ 8.0, 12.0 ] ) )
    assert isinstance( res, tor.Tensor )
    assert str( res.device ) == osc.device
    assert res.dtype == osc.dtype
    assert res.shape == ( 2, )
    assert tor.isfinite( res ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_ar_system_recursive_evaluation( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''AR system recursively evaluates correctly over the full sequence.'''
    # y[k] = 0.5*y[k-1] + x[k]
    result: tor.Tensor = SymbolicOscillator(
                ModelVarNames = [ "x", "y" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k-1]", "x[k]" ],
                theta = tor.tensor( [ 0.5, 1.0 ] ),
                OutputVarName = "y"
            ).Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert tor.allclose( result, tor.tensor( [ 1.0, 2.5, 4.25 ] ) )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 3, )
    assert tor.isfinite( result ).all()

  def test_MA_with_lag( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A moving-average system with a lag produces correct delayed output.'''
    # y[k] = theta0 * x[k-1]
    # The MA part uses internal storage; start with zero initial condition
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 3.0 ] )
        )
    result: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] ) ] )
    assert result.shape == ( 5, )
    assert tor.allclose( result, 3.0 * tor.tensor( [ 0.0, 1.0, 2.0, 3.0, 4.0 ] ) )
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_buffer_start_with_correct_storage( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Initial input storage is correctly used for the first output sample.'''
    # y[k] = theta0 * x[k-1]  and we set previous x[-1] = 10.0
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 3.0 ] )
        )
    # Set input storage with a single past value
    osc.set_InputStorage( tor.tensor( [ [ 10.0 ] ] ) ) # shape (1,1)
    result: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert result.shape == ( 3, )
    assert tor.allclose( result, 3.0 * tor.tensor( [ 10.0, 1.0, 2.0 ] ) )
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_AR_system( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''An auto-regressive system with feedback produces correct recursive output.'''
    # y[k] = 0.5 * y[k-1] + x[k]   (theta0=0.5, theta1=1.0)
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]", "x[k]" ],
            theta = tor.tensor( [ 0.5, 1.0 ] ),
            OutputVarName = "y"
        )
    # initial output storage y[-1] = 0.0 (default)
    # compute manually: y[0] = 0.5*0 + 1.0*1 = 1
    # y[1] = 0.5*1 + 2 = 2.5, y[2] = 0.5*2.5 + 3 = 4.25, y[3] = 0.5*4.25 + 4 = 6.125
    result = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 4, )
    assert tor.allclose( result, tor.tensor( [ 1.0, 2.5, 4.25, 6.125 ] ) )
    # output storage should have been updated to the last value
    assert osc.get_OutputStorage().shape == ( 1, )
    assert osc.get_OutputStorage().item() == 6.125
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )

  def test_pure_AR_with_custom_initial_output( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A pure AR system with custom initial output storage works correctly.'''
    # System: y[k] = 0.5 * y[k-1]  +  DsData[k]   (no input variables used)
    # We include a dummy input variable "x" to satisfy the non-empty Data requirement.
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],  # x is unused but keeps Data non-empty
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]" ],
            theta = tor.tensor( [ 0.5 ] ),
            OutputVarName = "y"
        )
    osc.set_OutputStorage( tor.tensor( [ 10.0 ] ) )
    # Data: three samples of dummy input (values ignored)
    dummy: tor.Tensor = tor.zeros( 4 )
    result: tor.Tensor = osc.Oscillate( Data = [ dummy ], DsData = tor.tensor( [ 0.0, 0.0, 0.0, 0.0 ] ) )
    assert result.shape == ( 4, )
    assert tor.allclose( result, tor.tensor( [ 5.0, 2.5, 1.25, 0.625 ] ) )
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_rational_system( self, basic_inputs ) -> None:
    '''A rational (denominator) system produces correct output.'''
    # y = theta0*x / (1 + theta1*x[k-1])   (theta0=2, theta1=1)
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "~/(x[k-1])" ],
            theta = tor.tensor( [ 2.0, 1.0 ] )
        )
    # No initial storage, x[-1] treated as 0 -> denominator = 1 + 1*0 = 1
    # k=0: y=2*1/1=2, k=1: y=2*2/(1+1*1)=4/2=2, k=2: y=2*3/(1+2)=6/3=2, k=3: y=2*4/(1+3)=8/4=2
    result = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 4, )
    assert close(
            result,
            tor.tensor( [ 2.0, 2.0, 2.0, 2.0 ] )
        )

  def test_positive_input_lag_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A positive lag on input variables raises ValueError during oscillation.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k+1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Oscillate( [ tor.randn( 10 ) ] )

  def test_zero_theta_modulation( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Theta can be overridden at Oscillate call time.'''
    # change theta on the fly
    x: tor.Tensor = tor.tensor( [ 5.0 ] )
    result: tor.Tensor = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ], theta = tor.tensor( [ 3.0 ] ) )
    assert result.shape == ( 1, )
    assert tor.allclose( result, 3.0 * x )
    assert isinstance( result, tor.Tensor )
    assert tor.isfinite( result ).all()

  def test_storage_propagation_across_buffers( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Internal storage propagates correctly across multiple Oscillate calls.'''
    # y[k] = x[k-1]
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    # first buffer
    result1: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert tor.allclose( result1, tor.tensor( [ 0.0, 1.0, 2.0 ] ) )
    # storage should now hold last value = 3.0
    assert osc.get_InputStorage().item() == 3.0
    # second buffer
    result2: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 4.0, 5.0 ] ) ] )
    assert tor.allclose( result2, tor.tensor( [ 3.0, 4.0 ] ) )
    assert isinstance( result1, tor.Tensor )
    assert isinstance( result2, tor.Tensor )
    assert tor.isfinite( result1 ).all()
    assert tor.isfinite( result2 ).all()
    assert str( result1.device ) == osc.device
    assert str( result2.device ) == osc.device
    assert result1.dtype == osc.dtype
    assert result2.dtype == osc.dtype
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_get_set_theta( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Getting and setting theta preserves the values.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "x[k-1]" ],
            theta = tor.ones( 2 )
        )
    new_theta: tor.Tensor = tor.tensor( [ 2.0, 3.0 ] )
    osc.set_theta( new_theta )
    assert tor.allclose( osc.get_theta(), new_theta )
    assert isinstance( osc.get_theta(), tor.Tensor )
    assert osc.get_theta().dtype == new_theta.dtype
    assert osc.get_theta().shape == new_theta.shape

  def test_set_input_storage_wrong_shape( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Setting input storage with the wrong shape raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "InputStorage has wrong dimension" ):
      osc.set_InputStorage( tor.randn( 3 ) ) # expected (1,1)

  def test_zero_internal_storage( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Zeroing internal storage resets it to zeros.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    osc.set_InputStorage( tor.tensor( [ [ 5.0 ] ] ) )
    osc.zeroInternalStorage()
    # storage should be back to zeros
    assert osc.get_InputStorage().item() == 0.0
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_output_storage_get_set( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Output storage can be set and retrieved correctly.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]" ],
            theta = tor.tensor( [ 1.0 ] ),
            OutputVarName = "y"
        )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    assert osc.get_OutputStorage().item() == 7.0
    assert isinstance( osc.get_OutputStorage(), tor.Tensor )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    assert osc.get_OutputStorage().item() == 7.0

  def test_empty_lag_range( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A system with no lags (instantaneous only) works with long inputs.'''
    # system with only instantaneous x[k] - no lags
    x: tor.Tensor = tor.randn( 50 )
    result: tor.Tensor = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 3.0 ] )
            ).Oscillate( [ x ] )
    assert result.shape == ( 50, )
    assert tor.allclose( result, 3.0 * x )
    assert isinstance( result, tor.Tensor )
    assert tor.isfinite( result ).all()

  def test_zero_lag_system_buffer_start_skipped( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A zero-lag system skips the buffer start phase correctly.'''
    # y[k] = x[k]  (MaxNegLag=0) - entire loop runs main phase only
    x: tor.Tensor = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    # No missing samples, output exactly x
    result: tor.Tensor = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ] )
    assert result.shape == ( 3, )
    assert tor.equal( result, x )
    assert isinstance( result, tor.Tensor )
    assert tor.isfinite( result ).all()
  def test_denominator_only_system( self, basic_inputs ) -> None:
    '''A system with only denominator terms (MA_NumExpr empty) works.'''
    # y = 1 / (1 + theta0 * x[k-1])   (MA_NumExpr empty -> NumExpr="1")
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "~/(x[k-1])" ],
            theta = tor.tensor( [ 0.5 ] )
        )
    osc.set_InputStorage( tor.tensor( [ [ 2.0 ] ] ) ) # x[-1]=2
    # k=0: 1/(1+0.5*2) = 1/2=0.5, k=1: 1/(1+0.5*1)=1/1.5=0.6667
    result = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 2, )
    assert tor.allclose(
            result,
            tor.tensor( [ 0.5, 2 / 3 ] ),
            rtol = 1e-5
        )

  def test_missing_DsData_shape( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''DsData with mismatched shape raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "dimension doesn't equal" ):
      osc.Oscillate( [ tor.randn( 10 ) ], DsData = tor.randn( 5 ) )

  def test_set_theta_wrong_type_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''set_theta with non-Tensor raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "theta must be a torch.Tensor" ):
      osc.set_theta( [ 2.0 ] )

  def test_set_theta_wrong_length_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''set_theta with wrong length raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "y[k-1]" ],
            theta = tor.tensor( [ 1.0, 1.0 ] ),
            OutputVarName = "y"
        )
    with pytest.raises( ValueError, match = "theta has wrong dimension" ):
      osc.set_theta( tor.ones( 3 ) )

  def test_Data_not_list_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Oscillate with non-list Data raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Data must be a list or tuple" ):
      osc.Oscillate( Data = "not_a_list" )

  def test_Data_empty_raises( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Oscillate with empty Data raises ValueError.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Data can't be empty" ):
      osc.Oscillate( Data = [] )

  def test_zero_internal_storage_AR( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Zeroing internal storage resets both input and output storage for AR systems.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]", "x[k-1]" ],
            theta = tor.tensor( [ 0.5, 1.0 ] ),
            OutputVarName = "y"
        )
    osc.set_InputStorage( tor.tensor( [ [ 3.0 ] ] ) )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    osc.zeroInternalStorage()
    assert osc.get_InputStorage().item() == 0.0
    assert osc.get_OutputStorage().item() == 0.0
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_Oscillate_output_device( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Oscillate output is on the correct device.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 2.0 ] ),
            device = "cpu"
        )
    result: tor.Tensor = osc.Oscillate( [ tor.randn( 10 ) ] )
    assert result.device.type == "cpu"
    assert isinstance( result, tor.Tensor )
    assert result.dtype == osc.dtype
    assert result.shape == ( 10, )
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_dsdata_injection_behavior( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''DsData is correctly injected into the output.'''
    x: tor.Tensor = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    ds: tor.Tensor = tor.tensor( [ 0.5, 0.5, 0.5 ] )
    result: tor.Tensor = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ], DsData = ds )
    assert result.shape == ( 3, )
    assert tor.allclose( result, x + ds )
    assert isinstance( result, tor.Tensor )
    assert tor.isfinite( result ).all()

  def test_rational_system( self, basic_inputs: list[ NonLinearity ] ) -> None:
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "~/(x[k-1])" ],
            theta = tor.tensor( [ 2.0, 1.0 ] )
        )
    result: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 4, )
    assert tor.allclose( result, tor.tensor( [ 2.0, 2.0, 2.0, 2.0 ] ) )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_denominator_only_system( self, basic_inputs: list[ NonLinearity ] ) -> None:
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "~/(x[k-1])" ],
            theta = tor.tensor( [ 0.5 ] )
        )
    osc.set_InputStorage( tor.tensor( [ [ 2.0 ] ] ) )
    result: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0 ] ) ] )
    assert result.shape == ( 2, )
    assert tor.allclose( result, tor.tensor( [ 0.5, 2 / 3 ] ), rtol = 1e-5 )
    assert isinstance( result, tor.Tensor )
    assert str( result.device ) == osc.device
    assert result.dtype == osc.dtype
    assert tor.isfinite( result ).all()
    assert osc.get_InputStorage() is not None
    assert osc.get_OutputStorage() is not None
    assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
    assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )

  def test_positive_lag_buffer_end_truncation( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Positive input lag raises error.'''
    # positive lag means the system needs future values; currently it raises an error
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k+2]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Oscillate( [ tor.randn( 20 ) ] )

  def test_rational_system_compilation( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''A rational system can be oscillated without error and produces correct values.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "~/(x[k-1])" ],
            theta = tor.tensor( [ 2.0, 1.0 ] )
        )
    result: tor.Tensor = osc.Oscillate( [ tor.randn( 5 ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape[ 0 ] == 5
    assert tor.isfinite( result ).all()

  def test_storage_setters_validation( self, basic_inputs: list[ NonLinearity ] ) -> None:
    '''Storage setters validate shape and store values correctly.'''
    osc: SymbolicOscillator = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    # Wrong shape
    with pytest.raises( ValueError, match = "InputStorage has wrong dimension" ):
      osc.set_InputStorage( tor.randn( 3 ) ) # should be (1, 1)
    # Correct shape
    osc.set_InputStorage( tor.tensor( [ [ 5.0 ] ] ) )
    assert osc.get_InputStorage().item() == 5.0
    assert isinstance( osc.get_InputStorage(), tor.Tensor )

# =============================================================================
# Nested nonlinear verification tests with STORAGE CROSSING.
# Values computed step-by-step in a verified Python session at test-creation.
# InputStorage ordering: [x[-L], ..., x[-1]]  (oldest to newest, shape (nInputs, MaxInputLag))
# OutputStorage ordering: [y[-L], ..., y[-1]]  (oldest to newest offset, shape (MaxOutputLag,))
# =============================================================================

def test_nested_x2_plus_xm2_storage_crossing() -> None:
  '''sin(cos(2*x + 3*x[k-2])) — MA only, lag-2 forces buffer crossing at k=0,1'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  x: tor.Tensor = tor.tensor( [ 3.0, 4.0, 5.0, 6.0 ] )
  expected: tor.Tensor = tor.tensor( [ -0.790197, 0.136312, 0.835315, 0.411573 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x" ], nlin, [ "sin(cos(2*x + 3*x[k-2]))" ], tor.ones( 1 ), dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 1.0, 2.0 ] ] ) ) # x[-2]=1.0, x[-1]=2.0
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 4, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 2
  assert osc.get_MaxOutputLag() == 0
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_ar_lag2_nested_nonlin_storage_crossing() -> None:
  '''sin(2*x[k-1] + cos(x[k] - 0.5*y[k-2])) — AR lag-1 input, lag-2 output, buffer-start k=0,1'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  x: tor.Tensor = tor.tensor( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] )
  expected: tor.Tensor = tor.tensor( [ 0.999941, 0.968675, -0.057215, -0.937032, 0.897383 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "sin(2*x[k-1] + cos(x[k] - 0.5*y[k-2]))" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 0.5 ] ] ) ) # x[-1]=0.5
  osc.set_OutputStorage( tor.tensor( [ 0.1, 0.5 ] ) ) # y[-2]=0.1, y[-1]=0.5
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 5, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 1
  assert osc.get_MaxOutputLag() == 2
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_deep_ar_lag3_nested_nonlin() -> None:
  '''sin(cos(x[k-1]) + 0.5*y[k-3]) — deep AR lag-3, buffer-start k=0,1,2'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  x: tor.Tensor = tor.tensor( [ 0.5, 1.0, 1.5, 2.0, 2.5 ] )
  expected: tor.Tensor = tor.tensor( [ 0.710566, 0.856052, 0.556612, 0.413250, 0.011879 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "sin(cos(x[k-1]) + 0.5*y[k-3])" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 1.0 ] ] ) ) # x[-1]=1.0
  osc.set_OutputStorage( tor.tensor( [ 0.5, 0.3, 0.1 ] ) ) # y[-3]=0.5, y[-2]=0.3, y[-1]=0.1
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 5, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 1
  assert osc.get_MaxOutputLag() == 3
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_abs_exp_xkm2_minus_ykm1() -> None:
  '''abs(exp(x[k-2]) - 0.3*y[k-1]) — MA lag-2 + AR lag-1 cross-boundary'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "abs", tor.abs ), NonLinearity( "exp", tor.exp ) ]
  x: tor.Tensor = tor.tensor( [ 0.6, 0.8, 1.0, 1.2 ] )
  expected: tor.Tensor = tor.tensor( [ 1.191403, 1.134404, 1.481798, 1.781002 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "abs(exp(x[k-2]) - 0.3*y[k-1])" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 0.2, 0.4 ] ] ) ) # x[-2]=0.2, x[-1]=0.4
  osc.set_OutputStorage( tor.tensor( [ 0.1 ] ) ) # y[-1]=0.1
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 4, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 2
  assert osc.get_MaxOutputLag() == 1
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_coeff_nested_ar_ma_crossing() -> None:
  '''2*sin(0.5*y[k-1] + cos(x[k-2])) — coefficient x nested AR+MA with lag-2 buffer crossing'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  x: tor.Tensor = tor.tensor( [ 2.0, 2.5, 3.0, 3.5 ] )
  expected: tor.Tensor = tor.tensor( [ 1.421132, 1.408411, 0.568182, -0.988640 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "2*sin(0.5*y[k-1] + cos(x[k-2]))" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 1.0, 1.5 ] ] ) ) # x[-2]=1.0, x[-1]=1.5
  osc.set_OutputStorage( tor.tensor( [ 0.5 ] ) ) # y[-1]=0.5
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 4, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 2
  assert osc.get_MaxOutputLag() == 1
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_storage_propagation_nested_cross_batches() -> None:
  '''sin(cos(x[k-1])) called twice — second batch uses storage from first'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  osc: SymbolicOscillator = SymbolicOscillator( [ "x" ], nlin, [ "sin(cos(x[k-1]))" ], tor.ones( 1 ), dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 0.0 ] ] ) ) # x[-1]=0.0
  r1: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 0.5, 1.0 ] ) ] )
  assert tor.allclose( r1, tor.tensor( [ 0.841471, 0.769196 ] ), atol = 1e-5 )
  # After first call, InputStorage holds x[-1] = 1.0
  r2: tor.Tensor = osc.Oscillate( [ tor.tensor( [ 1.5, 2.0 ] ) ] )
  assert tor.allclose( r2, tor.tensor( [ 0.514395, 0.070678 ] ), atol = 1e-5 )
  assert isinstance( r1, tor.Tensor )
  assert isinstance( r2, tor.Tensor )
  assert tor.isfinite( r1 ).all()
  assert tor.isfinite( r2 ).all()
  assert str( r1.device ) == osc.device
  assert str( r2.device ) == osc.device
  assert r1.dtype == osc.dtype
  assert r2.dtype == osc.dtype
  assert osc.get_MaxInputLag() == 1
  assert osc.get_MaxOutputLag() == 0
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_multi_term_nested_ar() -> None:
  '''sin(cos(x[k-1])) + 0.5*y[k-1] — multiple terms with AR recursion'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  x: tor.Tensor = tor.tensor( [ 0.2, 0.4, 0.6, 0.8 ] )
  expected: tor.Tensor = tor.tensor( [ 0.841471, 1.251270, 1.421879, 1.445715 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "sin(cos(x[k-1])) + 0.5*y[k-1]" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 0.0 ] ] ) )
  osc.set_OutputStorage( tor.tensor( [ 0.0 ] ) )
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 4, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 1
  assert osc.get_MaxOutputLag() == 1
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_sin_exp_xkm2_minus_ykm1() -> None:
  '''sin(exp(x[k-2]) - y[k-1]) — MA lag-2 + AR lag-1 with exp inside sin'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "exp", tor.exp ) ]
  x: tor.Tensor = tor.tensor( [ 0.5, 0.7, 0.9, 1.1, 1.3 ] )
  expected: tor.Tensor = tor.tensor( [ 0.844254, 0.484337, 0.918545, 0.889023, 1.000000 ] )
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "sin(exp(x[k-2]) - y[k-1])" ], tor.ones( 1 ), OutputVarName = "y", dtype = tor.float64 )
  osc.set_InputStorage( tor.tensor( [ [ 0.1, 0.3 ] ] ) ) # x[-2]=0.1, x[-1]=0.3
  osc.set_OutputStorage( tor.tensor( [ 0.1 ] ) ) # y[-1]=0.1
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert tor.allclose( result, expected, atol = 1e-5 )
  assert isinstance( result, tor.Tensor )
  assert str( result.device ) == osc.device
  assert result.dtype == osc.dtype
  assert result.shape == ( 5, )
  assert tor.isfinite( result ).all()
  assert osc.get_MaxInputLag() == 2
  assert osc.get_MaxOutputLag() == 1
  assert osc.get_nInputVars() == 1
  assert osc.get_nRegressors() == 1
  assert osc.get_InputStorage() is not None
  assert osc.get_OutputStorage() is not None
  assert osc.get_InputStorage().shape == ( osc.get_nInputVars(), osc.get_MaxInputLag() )
  assert osc.get_OutputStorage().shape == ( osc.get_MaxOutputLag(), )


def test_exponent_expression() -> None:
  '''(sin(0.5*x[k-1]))^(cos(0.3*y[k-1])) — parenthesized expression as exponent, binary ^ operator'''
  nlin: list[ NonLinearity ] = [ NonLinearity( "id", lambda x : x ), NonLinearity( "sin", tor.sin ), NonLinearity( "cos", tor.cos ) ]
  osc: SymbolicOscillator = SymbolicOscillator( [ "x", "y" ], nlin, [ "( sin( 0.5*x[k-1] ) ) ^ ( cos( 0.3*y[k-1] ) )" ],
                             tor.tensor( [ 1.0 ] ), OutputVarName = "y", dtype = tor.float64 )
  x: tor.Tensor = tor.tensor( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] )
  expected: tor.Tensor = tor.tensor( [ 0.0, 0.479425538604203, 0.8429719909481942, 0.9975745669116292, 0.9131485415825158 ] )
  result: tor.Tensor = osc.Oscillate( [ x ] )
  assert isinstance( result, tor.Tensor )
  assert tor.isfinite( result ).all()
  assert result.shape == x.shape
  assert tor.allclose( result, expected, atol = 1e-5 )
