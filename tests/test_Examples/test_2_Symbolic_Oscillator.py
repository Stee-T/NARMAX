import numpy as np
import torch as tor
import matplotlib
matplotlib.use( "Agg" )

import NARMAX
import pytest


def UserFunction( x: tor.Tensor ) -> tor.Tensor:
    return tor.max( tor.tensor( [ 0.0 ] ), tor.sin( 2 * x ) )


InputVarNames: list[ str ] = [ 'x1', 'x2', 'y' ]
NonLinearities: list[ NARMAX.NonLinearity ] = [
    NARMAX.Identity,
    NARMAX.NonLinearity( "uFunc", f = UserFunction ),
    NARMAX.NonLinearity( "abs", f = tor.abs ),
    NARMAX.NonLinearity( "cos", f = tor.cos ),
]
Expressions: list[ str ] = [
    "y[k-1]/x2[k]",
    "uFunc( x1[k-1] )",
    "1/abs( 0.2*x1[k-1] + 0.5*x1[k-2]*x2[k] - 0.2 )",
    "~/(x1[k-1]*x2[k-1])",
    "~/(x2[k-2]^2)",
    "~/cos( 0.2*x1[k-3]*x2[k-1] - 0.1 )"
]


def test_symbolic_oscillator_constructor_validation() -> None:
    '''Test SymbolicOscillator constructor rejects invalid inputs.'''
    theta = tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] )

    with pytest.raises( ValueError, match = "No Input variables were declared" ):
        NARMAX.SymbolicOscillator( [], NonLinearities, Expressions, theta )
    with pytest.raises( ValueError, match = "Duplicate names found in ModelVarNames" ):
        NARMAX.SymbolicOscillator( [ 'x1', 'x1' ], NonLinearities[ :1 ], [ "x1[k]" ], theta[ :1 ] )
    with pytest.raises( ValueError, match = "No Non-linearities were declared" ):
        NARMAX.SymbolicOscillator( [ 'x1' ], [], [ "x1[k]" ], theta[ :1 ] )
    with pytest.raises( ValueError, match = "No Regressors were declared" ):
        NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, [], theta )
    with pytest.raises( ValueError, match = "theta must be of type 'torch.Tensor'" ):
        NARMAX.SymbolicOscillator( [ 'x1' ], NonLinearities[ :1 ], [ "x1[k]" ], [ 1.0 ] )
    with pytest.raises( ValueError, match = "theta length" ):
        NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[ :1 ] )
    with pytest.raises( ValueError, match = "Duplicate non-linearity names found" ):
        NARMAX.SymbolicOscillator( [ 'x1' ], [ NonLinearities[ 0 ], NonLinearities[ 0 ] ], [ "x1[k]" ], theta[ :1 ] )


def test_symbolic_oscillator_properties() -> None:
    '''Test SymbolicOscillator getter methods return correct values.'''
    theta = tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] )
    model = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta )

    assert model.get_nRegressors() == 6
    assert model.get_nInputVars() == 2
    assert model.get_MaxInputLag() == 3
    assert model.get_MaxOutputLag() == 1
    assert model.get_MaxPositiveInputLag() == 0

    got_theta = model.get_theta()
    assert isinstance( got_theta, tor.Tensor )
    assert got_theta.shape == theta.shape
    assert tor.allclose( got_theta.cpu(), theta.cpu() )


def test_symbolic_oscillator_set_theta() -> None:
    '''Test theta setter/getter with valid and invalid inputs.'''
    theta = tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] )
    model = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta )

    new_theta = tor.tensor( [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] )
    model.set_theta( new_theta )
    assert tor.allclose( model.get_theta(), new_theta )

    with pytest.raises( ValueError, match = "theta must be a torch.Tensor" ):
        model.set_theta( [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] )
    with pytest.raises( ValueError, match = "theta has wrong dimension" ):
        model.set_theta( tor.tensor( [ 1.0, 2.0 ] ) )


def test_symbolic_oscillator_storage() -> None:
    '''Test internal storage getters/setters and zeroInternalStorage.'''
    theta = tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] )
    model = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta )

    init_input = model.get_InputStorage()
    assert isinstance( init_input, tor.Tensor )
    assert init_input.shape == ( 2, 3 )
    assert tor.allclose( init_input, tor.zeros( 2, 3 ) )

    init_output = model.get_OutputStorage()
    assert isinstance( init_output, tor.Tensor )
    assert init_output.shape == ( 1, )
    assert tor.allclose( init_output, tor.zeros( 1 ) )

    new_input = tor.tensor( [ [ 1.0, 2.0, 3.0 ], [ 4.0, 5.0, 6.0 ] ] )
    model.set_InputStorage( new_input )
    assert tor.allclose( model.get_InputStorage().cpu(), new_input.cpu() )

    new_output = tor.tensor( [ 7.0 ] )
    model.set_OutputStorage( new_output )
    assert tor.allclose( model.get_OutputStorage().cpu(), new_output.cpu() )

    with pytest.raises( ValueError, match = "wrong dimension" ):
        model.set_InputStorage( tor.tensor( [ 1.0, 2.0 ] ) )
    with pytest.raises( ValueError, match = "wrong dimension" ):
        model.set_OutputStorage( tor.tensor( [ 1.0, 2.0 ] ) )

    model.zeroInternalStorage()
    assert tor.allclose( model.get_InputStorage(), tor.zeros( 2, 3 ) )
    assert tor.allclose( model.get_OutputStorage(), tor.zeros( 1 ) )


def test_symbolic_oscillator_oscillate_validation() -> None:
    '''Test Oscillate method rejects invalid arguments.'''
    theta = tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] )
    model = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta )
    x = tor.randn( 100 )

    with pytest.raises( ValueError, match = "Data must be a list or tuple" ):
        model.Oscillate( Data = x )
    with pytest.raises( ValueError, match = "DsData must be a torch.tensor or None" ):
        model.Oscillate( Data = [ x, x ], DsData = [ 1.0 ] )
    with pytest.raises( ValueError, match = "Data can't be empty, nothing to oscillate" ):
        model.Oscillate( Data = [] )
    with pytest.raises( ValueError, match = "DsData must be 1D" ):
        model.Oscillate( Data = [ x, x ], DsData = tor.randn( 10, 2 ) )
    with pytest.raises( ValueError, match = "DsData's dimension doesn't equal that of Data" ):
        model.Oscillate( Data = [ x, x ], DsData = tor.randn( 50 ) )


def test_user_function() -> None:
    '''Test the UserFunction used in the oscillator system.'''
    assert isinstance( UserFunction( tor.tensor( [ 0.0 ] ) ), tor.Tensor )

    result = UserFunction( tor.tensor( [ 0.0 ] ) )
    assert result.item() == 0.0

    result = UserFunction( tor.tensor( [ -1.0 ] ) )
    assert result.item() == 0.0

    x = tor.tensor( [ tor.pi / 4 ] )
    result = UserFunction( x )
    assert result.item() > 0.0
    assert tor.allclose( result, tor.sin( 2 * x ) )

    result = UserFunction( tor.tensor( [ tor.pi ] ) )
    assert result.item() == 0.0


def test_nonlinearity_configuration() -> None:
    '''Test the NonLinearity objects used in the oscillator.'''
    assert isinstance( NARMAX.Identity, NARMAX.NonLinearity )
    assert NARMAX.Identity.get_Name() == "id"

    names = [ nl.get_Name() for nl in NonLinearities ]
    assert names == [ "id", "uFunc", "abs", "cos" ]
    assert len( names ) == len( set( names ) )

    for nl in NonLinearities:
        assert isinstance( nl.get_f()( tor.randn( 10 ) ), tor.Tensor )


def test_symbolic_oscillator() -> None:
    '''Integration test for SymbolicOscillator against a reference system.'''
    tor.manual_seed( 123 )

    # --------------------------------------------------- 1. Data Generation
    p: int = 2_000
    InputAmplitude: float = 1.5

    x1: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
    x2: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )

    def System( y, x1, x2, W, theta, StartIdx, EndIdx ):
        for k in range( StartIdx, EndIdx ):
            num = ( theta[ 0 ] * y[ k - 1 ] / x2[ k ] +
                         theta[ 1 ] * UserFunction( x1[ k - 1 ] ) +
                         theta[ 2 ] / tor.abs( 0.2 * x1[ k - 1 ] + 0.5 * x1[ k - 2 ] * x2[ k ] - 0.2 ) )
            den = ( 1 + theta[ 3 ] * x1[ k - 1 ] * x2[ k - 1 ] +
                         theta[ 4 ] * x2[ k - 2 ]**2 +
                         theta[ 5 ] * tor.cos( 0.2 * x1[ k - 3 ] * x2[ k - 1 ] - 0.1 ) )
            y[ k ] = W[ k ] + num / den
        return y

    AdditionalInput = 0.2 * ( tor.rand( p ) - 0.5 )

    theta = [
        tor.tensor( [ 0.2, -0.3, 1.0, 0.8, -0.3, 1.0 ] ),
        tor.tensor( [ 0.25, -0.25, 0.8, 0.9, -0.5, 0.95 ] ),
        tor.tensor( [ 0.3, -0.4, 0.7, 0.7, -0.4, 0.9 ] )
    ]

    ThirdBuffer = p // 3

    # --------------------------------------------------- 2. Symbolic Oscillator Data
    # --------------------------------------------------- 3. Simulation and Processing
    # y = Real System
    y = tor.zeros( p )
    for i in range( 3 ):
        StartIdx = ( i * ThirdBuffer ) if i > 0 else 3
        EndIdx = ( i + 1 ) * ThirdBuffer
        y = System( y, x1, x2, AdditionalInput, theta[ i ], StartIdx, EndIdx )

    # yHat = Symbolic Oscillator
    Model = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[ 0 ] )
    yHat = tor.zeros( p )
    for i in range( 3 ):
        StartIdx = i * ThirdBuffer
        EndIdx = ( i + 1 ) * ThirdBuffer
        out = Model.Oscillate(
            Data = [ x1[ StartIdx : EndIdx ], x2[ StartIdx : EndIdx ] ],
            theta = theta[ i ],
            DsData = AdditionalInput[ StartIdx : EndIdx ]
        )
        assert isinstance( out, tor.Tensor )
        assert out.shape == ( EndIdx - StartIdx, )
        assert out.dtype == tor.get_default_dtype()
        yHat[ StartIdx : EndIdx ] = out

    # --------------------------------------------------- 4. Error Analysis (assertion)
    diff = ( y - yHat )[ 20 : ]
    assert isinstance( diff, tor.Tensor )
    assert diff.ndim == 1
    assert diff.shape[ 0 ] == p - 20

    assert Model.get_nRegressors() == 6
    assert Model.get_nInputVars() == 2
    assert Model.get_MaxInputLag() == 3
    assert Model.get_MaxOutputLag() == 1
    assert tor.allclose( Model.get_theta().cpu(), theta[ -1 ].cpu() )

    np.testing.assert_allclose(
        diff.detach().cpu().numpy(),
        0.0,
        atol = 1e-5,
        rtol = 1e-5,
        err_msg = "SymbolicOscillator output deviates from reference system"
    )
