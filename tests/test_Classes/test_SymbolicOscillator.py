import pytest
import torch as tor

# --- Adjust these imports to your real module layout ---
import NARMAX
from NARMAX.Classes.Parser_0_2 import ExpressionParser, CleanExpression, SubExpression, ParsedReg
from NARMAX.Classes.SymbolicOscillator_0_3 import SymbolicOscillator
from NARMAX.Classes.NonLinearity import NonLinearity

# ###############################################################################
# 1. SymbolicOscillator construction & edge cases
# ###############################################################################
def make_identity_nonlin() -> NonLinearity: return NonLinearity( "id", lambda x: x )


@pytest.fixture
def basic_inputs() -> list:
  # simple non-linearities
  nl = [
        NonLinearity( "id", lambda x: x ),
        NonLinearity( "abs", tor.abs ),
    ]
  return nl


class TestConstructor:
  def test_valid_construction( self, basic_inputs ) -> None:
    '''Constructing a valid SymbolicOscillator works correctly.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x1", "x2" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x1", "x2[k-1]", "id(x1[k])" ],
            theta = tor.ones( 3 )
        )
    nreg = osc.get_nRegressors()
    assert isinstance( nreg, int )
    assert nreg == 3
    nvars = osc.get_nInputVars()
    assert isinstance( nvars, int )
    assert nvars == 2
    assert osc.get_nInputVars() == 2
    assert close( osc.get_theta(), tor.ones( 3 ) )
    assert osc.get_MaxInputLag() == 1  # from "x2[k-1]"
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

  def test_missing_nonlinearity_raises( self, basic_inputs ) -> None:
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

  def test_empty_ModelVarNames_raises( self, basic_inputs ) -> None:
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

  def test_reserved_name_OutVec_variable_raises( self, basic_inputs ) -> None:
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
def close( a: tor.Tensor, b: tor.Tensor, rtol: float = 1e-5 ) -> bool: return tor.allclose( a, b, rtol = rtol )


class TestOscillate:
  def test_simple_MA_system( self, basic_inputs ) -> None:
    '''A simple moving-average system (no lags) produces correct output.'''
    # y[k] = theta0 * x[k]   (no lags)
    x = tor.randn( 100 )
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 2.0 ] )
        )
    result = osc.Oscillate( [ x ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 100, )
    assert result.dtype == x.dtype
    assert close( result, 2.0 * x )

  def test_MA_with_lag( self, basic_inputs ) -> None:
    '''A moving-average system with a lag produces correct delayed output.'''
    # y[k] = theta0 * x[k-1]
    # The MA part uses internal storage; start with zero initial condition
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 3.0 ] )
        )
    result = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 5, )
    assert close(
            result,
            3.0 * tor.tensor( [ 0.0, 1.0, 2.0, 3.0, 4.0 ] )
        )

  def test_buffer_start_with_correct_storage( self, basic_inputs ) -> None:
    '''Initial input storage is correctly used for the first output sample.'''
    # y[k] = theta0 * x[k-1]  and we set previous x[-1] = 10.0
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 3.0 ] )
        )
    # Set input storage with a single past value
    osc.set_InputStorage( tor.tensor( [ [ 10.0 ] ] ) ) # shape (1,1)
    result = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 3, )
    assert close(
            result,
            3.0 * tor.tensor( [ 10.0, 1.0, 2.0 ] )
        )

  def test_AR_system( self, basic_inputs ) -> None:
    '''An auto-regressive system with feedback produces correct recursive output.'''
    # y[k] = 0.5 * y[k-1] + x[k]   (theta0=0.5, theta1=1.0)
    osc = SymbolicOscillator(
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
    assert close(
            result,
            tor.tensor( [ 1.0, 2.5, 4.25, 6.125 ] )
        )
    # output storage should have been updated to the last value
    assert osc.get_OutputStorage().shape == ( 1, )
    assert osc.get_OutputStorage().item() == 6.125

  def test_pure_AR_with_custom_initial_output( self, basic_inputs ) -> None:
    '''A pure AR system with custom initial output storage works correctly.'''
    # System: y[k] = 0.5 * y[k-1]  +  DsData[k]   (no input variables used)
    # We include a dummy input variable "x" to satisfy the non-empty Data requirement.
    osc = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ], # x is unused but keeps Data non-empty
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]" ],
            theta = tor.tensor( [ 0.5 ] ),
            OutputVarName = "y"
        )
    osc.set_OutputStorage( tor.tensor( [ 10.0 ] ) )
    # Data: three samples of dummy input (values ignored)
    dummy = tor.zeros( 4 )
    result = osc.Oscillate( Data = [ dummy ], DsData = tor.tensor( [ 0.0, 0.0, 0.0, 0.0 ] ) )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 4, )
    assert tor.allclose( result, tor.tensor( [ 5.0, 2.5, 1.25, 0.625 ] ) )

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

  def test_positive_input_lag_raises( self, basic_inputs ) -> None:
    '''A positive lag on input variables raises ValueError during oscillation.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k+1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Oscillate( [ tor.randn( 10 ) ] )

  def test_zero_theta_modulation( self, basic_inputs ) -> None:
    '''Theta can be overridden at Oscillate call time.'''
    # change theta on the fly
    x = tor.tensor( [ 5.0 ] )
    result = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ], theta = tor.tensor( [ 3.0 ] ) )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 1, )
    assert close(
            result,
            3.0 * x
        )

  def test_storage_propagation_across_buffers( self, basic_inputs ) -> None:
    '''Internal storage propagates correctly across multiple Oscillate calls.'''
    # y[k] = x[k-1]
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    # first buffer
    result1 = osc.Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert isinstance( result1, tor.Tensor )
    assert close( result1, tor.tensor( [ 0.0, 1.0, 2.0 ] ) )
    # storage should now hold last value = 3.0
    assert osc.get_InputStorage().item() == 3.0
    # second buffer
    result2 = osc.Oscillate( [ tor.tensor( [ 4.0, 5.0 ] ) ] )
    assert isinstance( result2, tor.Tensor )
    assert close(
            result2,
            tor.tensor( [ 3.0, 4.0 ] )
        )

  def test_get_set_theta( self, basic_inputs ) -> None:
    '''Getting and setting theta preserves the values.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "x[k-1]" ],
            theta = tor.ones( 2 )
        )
    new_theta = tor.tensor( [ 2.0, 3.0 ] )
    osc.set_theta( new_theta )
    assert close( osc.get_theta(), new_theta )

  def test_set_input_storage_wrong_shape( self, basic_inputs ) -> None:
    '''Setting input storage with the wrong shape raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "InputStorage has wrong dimension" ):
      osc.set_InputStorage( tor.randn( 3 ) ) # expected (1,1)

  def test_zero_internal_storage( self, basic_inputs ) -> None:
    '''Zeroing internal storage resets it to zeros.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    osc.set_InputStorage( tor.tensor( [ [ 5.0 ] ] ) )
    osc.zeroInternalStorage()
    # storage should be back to zeros
    assert osc.get_InputStorage().item() == 0.0

  def test_output_storage_get_set( self, basic_inputs ) -> None:
    '''Output storage can be set and retrieved correctly.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "y[k-1]" ],
            theta = tor.tensor( [ 1.0 ] ),
            OutputVarName = "y"
        )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    assert osc.get_OutputStorage().item() == 7.0

  def test_empty_lag_range( self, basic_inputs ) -> None:
    '''A system with no lags (instantaneous only) works with long inputs.'''
    # system with only instantaneous x[k] - no lags
    x = tor.randn( 50 )
    result = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 3.0 ] )
            ).Oscillate( [ x ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 50, )
    assert result.dtype == x.dtype
    assert tor.allclose(
            result,
            3.0 * x
        )

  def test_zero_lag_system_buffer_start_skipped( self, basic_inputs ) -> None:
    '''A zero-lag system skips the buffer start phase correctly.'''
    # y[k] = x[k]  (MaxNegLag=0) - entire loop runs main phase only
    x = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    # No missing samples, output exactly x
    result = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 3, )
    assert tor.equal(
            result,
            x
        )

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

  def test_missing_DsData_shape( self, basic_inputs ) -> None:
    '''DsData with mismatched shape raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "dimension doesn't equal" ):
      osc.Oscillate( [ tor.randn( 10 ) ], DsData = tor.randn( 5 ) )

  def test_set_theta_wrong_type_raises( self, basic_inputs ) -> None:
    '''set_theta with non-Tensor raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "theta must be a torch.Tensor" ):
      osc.set_theta( [ 2.0 ] )

  def test_set_theta_wrong_length_raises( self, basic_inputs ) -> None:
    '''set_theta with wrong length raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "y[k-1]" ],
            theta = tor.tensor( [ 1.0, 1.0 ] ),
            OutputVarName = "y"
        )
    with pytest.raises( ValueError, match = "theta has wrong dimension" ):
      osc.set_theta( tor.ones( 3 ) )

  def test_Data_not_list_raises( self, basic_inputs ) -> None:
    '''Oscillate with non-list Data raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Data must be a list or tuple" ):
      osc.Oscillate( Data = "not_a_list" )

  def test_Data_empty_raises( self, basic_inputs ) -> None:
    '''Oscillate with empty Data raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Data can't be empty" ):
      osc.Oscillate( Data = [] )

  def test_zero_internal_storage_AR( self, basic_inputs ) -> None:
    '''Zeroing internal storage resets both input and output storage for AR systems.'''
    osc = SymbolicOscillator(
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

  def test_Oscillate_output_device( self, basic_inputs ) -> None:
    '''Oscillate output is on the correct device.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x" ],
            theta = tor.tensor( [ 2.0 ] ),
            device = "cpu"
        )
    result = osc.Oscillate( [ tor.randn( 10 ) ] )
    assert result.device.type == "cpu"


# ###############################################################################
# 4. Internal function tests (Toggle, EvalStr, etc.)
# ###############################################################################
class TestInternals:
  def test_parsed_reg_to_eval_string( self ) -> None:
    '''ParsedReg2EvalStr produces correct evaluation string from a ParsedReg.'''
    # correct import path
    from NARMAX.Classes.SymbolicOscillator_0_3 import ParsedReg2EvalStr
    from NARMAX.Classes.Parser_0_2 import SubExpression, ParsedReg
    eval_str, is_ar = ParsedReg2EvalStr(
            ParsedReg(
                FuncName = None,
                SubExpressions = [ SubExpression( Coeff = 2.0, VarName = "x1", Lag = -1 ) ],
                Operators = []
            ),
            { "x1": 0 },
            {},
            OutputVarName = 'y'
        )
    assert "Data[0][k-1]" in eval_str
    assert is_ar == False

  def test_parsed_reg_to_eval_string_AR( self ) -> None:
    '''ParsedReg2EvalStr correctly identifies AR expressions.'''
    from NARMAX.Classes.SymbolicOscillator_0_3 import ParsedReg2EvalStr
    from NARMAX.Classes.Parser_0_2 import SubExpression, ParsedReg
    eval_str, is_ar = ParsedReg2EvalStr(
            ParsedReg(
                FuncName = None,
                SubExpressions = [ SubExpression( Coeff = 0.5, VarName = "y", Lag = -1 ) ],
                Operators = []
            ),
            { "x1": 0 },
            {},
            OutputVarName = 'y'
        )
    assert "OutVec[k-1]" in eval_str
    assert is_ar == True

  def test_parsed_reg_to_eval_string_with_nonlin( self ) -> None:
    '''ParsedReg2EvalStr with a non-linearity function produces correct string.'''
    from NARMAX.Classes.SymbolicOscillator_0_3 import ParsedReg2EvalStr
    from NARMAX.Classes.Parser_0_2 import SubExpression, ParsedReg
    eval_str, is_ar = ParsedReg2EvalStr(
            ParsedReg(
                FuncName = "abs",
                SubExpressions = [ SubExpression( VarName = "x1", Lag = 0 ) ],
                Operators = []
            ),
            { "x1": 0 },
            { "abs": 0 },
            OutputVarName = 'y'
        )
    assert "NonLinList[0].get_f()" in eval_str
    assert "Data[0][k+0]" in eval_str
    assert is_ar == False

  def test_parsed_reg_to_eval_string_with_operators( self ) -> None:
    '''ParsedReg2EvalStr with multiple sub-expressions and operators.'''
    from NARMAX.Classes.SymbolicOscillator_0_3 import ParsedReg2EvalStr
    from NARMAX.Classes.Parser_0_2 import SubExpression, ParsedReg
    eval_str, is_ar = ParsedReg2EvalStr(
            ParsedReg(
                FuncName = None,
                SubExpressions = [ SubExpression( VarName = "x1", Lag = 0 ), SubExpression( VarName = "x2", Lag = -1 ) ],
                Operators = [ "*" ]
            ),
            { "x1": 0, "x2": 1 },
            {},
            OutputVarName = 'y'
        )
    assert "Data[0][k+0]" in eval_str
    assert "Data[1][k-1]" in eval_str
    assert "*" in eval_str
    assert is_ar == False

  def test_buffer_toggle( self, basic_inputs ) -> None:
    '''Buffer_Toggle returns correct data using internal storage.'''
    # Test Buffer_Toggle directly
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    # Set input storage with a known past value
    osc.set_InputStorage( tor.tensor( [ [ 10.0 ] ] ) )
    # Buffer_Toggle(0, -1, data) should return [10.0, 1.0]
    assert close(
            osc.Buffer_Toggle( 0, -1, [ tor.tensor( [ 1.0, 2.0 ] ) ] ),
            tor.tensor( [ 10.0, 1.0 ] )
        )

  def test_buffer_toggle_k0( self, basic_inputs ) -> None:
    '''Buffer_Toggle with k=0 returns Data directly.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    data = [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ]
    assert close(
            osc.Buffer_Toggle( 0, 0, data ),
            data[ 0 ]
        )

  def test_buffer_toggle_positive_k_raises( self, basic_inputs ) -> None:
    '''Buffer_Toggle with k > 0 raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Buffer_Toggle( 0, 1, [ tor.randn( 5 ) ] )

  def test_scalar_toggle( self, basic_inputs ) -> None:
    '''Scalar_Toggle returns correct scalar using storage or output vector.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x", "y" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]", "y[k-1]" ],
            theta = tor.tensor( [ 1.0, 1.0 ] ),
            OutputVarName = "y"
        )
    osc.set_InputStorage( tor.tensor( [ [ 5.0 ] ] ) )
    osc.set_OutputStorage( tor.tensor( [ 7.0 ] ) )
    data = [ tor.tensor( [ 1.0 ] ), tor.tensor( [ 2.0 ] ) ]
    # Scalar_Toggle for input, k=-1 -> storage
    assert osc.Scalar_Toggle( 0, 0, -1, data ) == 5.0
    # for output, k=0 -> OutVec (not yet initialized, but we can test k>=0 with a temporary OutVec)
    osc.OutVec[ 0 ] = 3.0
    assert osc.Scalar_Toggle( 1, None, 0, data ) == 3.0
    # for input, k=0 -> current data
    assert osc.Scalar_Toggle( 0, 0, 0, data ) == 1.0
    # for output, k=-1 -> OutputStorage
    assert osc.Scalar_Toggle( 1, None, -1, data ) == 7.0

  def test_scalar_toggle_invalid_arg_raises( self, basic_inputs ) -> None:
    '''Scalar_Toggle with invalid DataOrOutVec raises ValueError.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k-1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Internal Error" ):
      osc.Scalar_Toggle( 2, None, 0, [ tor.randn( 1 ) ] )


# ###############################################################################
# 5. More tests
# ###############################################################################
class TestSymbolicOscillatorCore:

  def test_positive_input_lag_buffer_toggle( self, basic_inputs ) -> None:
    '''Positive input lag raises ValueError in Buffer_Toggle.'''
    # system with x[k+1] - positive lag triggers Buffer_Toggle error
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k+1]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Oscillate( [ tor.randn( 10 ) ] )

  def test_positive_lag_buffer_end_truncation( self, basic_inputs ) -> None:
    '''Positive input lag raises error.'''
    # positive lag means the system needs future values; currently it raises an error
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x[k+2]" ],
            theta = tor.tensor( [ 1.0 ] )
        )
    with pytest.raises( ValueError, match = "Lag must be negative or zero" ):
      osc.Oscillate( [ tor.randn( 20 ) ] )

  def test_rational_system_compilation( self, basic_inputs ) -> None:
    '''A rational system can be oscillated without error and produces correct values.'''
    osc = SymbolicOscillator(
            ModelVarNames = [ "x" ],
            NonLinearities = basic_inputs,
            ExprList = [ "x", "~/(x[k-1])" ],
            theta = tor.tensor( [ 2.0, 1.0 ] )
        )
    result = osc.Oscillate( [ tor.randn( 5 ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape[ 0 ] == 5

  def test_dsdata_injection_behavior( self, basic_inputs ) -> None:
    '''DsData is correctly injected into the output.'''
    x = tor.tensor( [ 1.0, 2.0, 3.0 ] )
    ds = tor.tensor( [ 0.5, 0.5, 0.5 ] )
    result = SymbolicOscillator(
                ModelVarNames = [ "x" ],
                NonLinearities = basic_inputs,
                ExprList = [ "x" ],
                theta = tor.tensor( [ 1.0 ] )
            ).Oscillate( [ x ], DsData = ds )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 3, )
    assert tor.allclose(
            result,
            x + ds
        )

  def test_storage_setters_validation( self, basic_inputs ) -> None:
    '''Storage setters validate shape and store values correctly.'''
    osc = SymbolicOscillator(
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

  def test_ar_system_recursive_evaluation( self, basic_inputs ) -> None:
    '''AR system recursively evaluates correctly over the full sequence.'''
    # y[k] = 0.5*y[k-1] + x[k]
    # manual: y[0]=0.5*0+1=1, y[1]=0.5*1+2=2.5, y[2]=0.5*2.5+3=4.25
    result = SymbolicOscillator(
                ModelVarNames = [ "x", "y" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k-1]", "x[k]" ],
                theta = tor.tensor( [ 0.5, 1.0 ] ),
                OutputVarName = "y"
            ).Oscillate( [ tor.tensor( [ 1.0, 2.0, 3.0 ] ) ] )
    assert isinstance( result, tor.Tensor )
    assert result.shape == ( 3, )
    assert tor.allclose(
            result,
            tor.tensor( [ 1.0, 2.5, 4.25 ] )
        )

  def test_non_causal_output_lag_rejection( self, basic_inputs ) -> None:
    '''Non-causal (positive) output lag is rejected at construction.'''
    with pytest.raises( ValueError, match = "Positive lag for output" ):
      SymbolicOscillator(
                ModelVarNames = [ "x", "y" ],
                NonLinearities = basic_inputs,
                ExprList = [ "y[k+1]" ],
                theta = tor.tensor( [ 1.0 ] ),
                OutputVarName = "y"
            )
