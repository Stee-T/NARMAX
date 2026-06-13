# ################################################################################ Example 7: ODE Discovery with DerivativeAugment and Mollifiers ################################################################################
# This tutorial demonstrates how to use `DiffedMollifiersCTor` and `DerivativeAugment` to discover
# ordinary differential equations from input-output data. We recover the parameters of a coupled
# spring-mass-damper system using only position measurements and the known input force.
#
# Key idea: Signal derivatives are HPF, thus to avoid exploding the HF content, one first
# smoothes the signal (LPF) to stabilize the estimation with mollifiers -- smooth functions
# used in distribution theory to create sequences of smooth functions approximating non-smooth
# functions via convolution. Being almost Dirac distributions (single lobes with integral=1),
# they're the convolution's identity element. So we convolve the measurements with differentiated
# Gaussians to get smoothed derivative estimates, then use those as regressors to discover the ODE.

# ---------------------------------------------------- 1. Imports
import torch as tor
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

import NARMAX
from NARMAX.CTors import DiffedGaussianMollifier, SmoothDeriver

def _sanitize( name: str ) -> str:
  '''Strip unicode from library-generated names for terminal-safety.'''
  _map: dict[ int, int ] = {
        0x2202: ord( 'd' ), # ∂ -> d
        0x00B9: ord( '1' ), # ¹ -> 1
        0x00B2: ord( '2' ), # ² -> 2
        0x00B3: ord( '3' ), # ³ -> 3
        0x2074: ord( '4' ), # ⁴ -> 4
        0x2075: ord( '5' ), # ⁵ -> 5
        0x2076: ord( '6' ), # ⁶ -> 6
        0x2077: ord( '7' ), # ⁷ -> 7
        0x2078: ord( '8' ), # ⁸ -> 8
        0x2079: ord( '9' ), # ⁹ -> 9
        0x2070: ord( '0' ), # ⁰ -> 0
    }
  return name.translate( _map )

# ---------------------------------------------------- 2. Hyper-parameters
p: int = 3_000 # Dataset size (number of samples)
Fs: float = 100.0 # Sampling frequency [Hz]
dt: float = 1.0 / Fs # Time step [s]
T: float = p * dt # Total simulation time [s]

# True system parameters (coupled spring-mass-damper -- ground truth for comparison)
m1: float = 1.0;  m2: float = 1.0 # Masses [kg]
k1: float = 100.0; k2: float = 50.0 # Stiffnesses [N/m]
c1: float = 2.0;   c2: float = 1.0 # Damping coefficients [Ns/m]

# DerivativeAugment parameters (explained in Section 3)
FilterOrder: int = 31 # FIR filter length (odd = peak on a sample, bigger = wider = more smoothing)
nDerivatives: int = 2 # How many derivative orders to compute (0 = identity, 1 = velocity, 2 = acceleration)
Std: float = 0.12 # Gaussian standard deviation (wider = more smoothing, narrower = less)
RegCoeffs: list[ float ] = [ 1.0, 1.0 ] # Per-derivative weights (acts like regularization if < 1)

# ---------------------------------------------------- 3. Differentiated Mollifiers -- Intuition and Parameter Tour
print( "\n" + "#" * 80 )
print( "  Section 3: Differentiated Mollifiers -- Intuition and Parameters" )
print( "#" * 80 + "\n" )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3a. Default filter shapes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  3a. Default DiffedMollifiersCTor output:" )

CoeffsDef: list[ tor.Tensor ] = DiffedGaussianMollifier( FilterOrder = FilterOrder, nDerivatives = nDerivatives,
                                                     std = Std, Plot = False )
print( f"     FilterOrder={ FilterOrder }, nDerivatives={ nDerivatives }, std={ Std }" )
print( f"     { len( CoeffsDef ) } filters returned (0th-order Gaussian + { nDerivatives } derivatives)" )
print( f"     0th-order sum = { tor.sum( CoeffsDef[ 0 ] ):.6f}  (L1 normalized -> ~ 1.0)" )
print( f"     1st-order sum = { tor.sum( CoeffsDef[ 1 ] ):.6f}  (antisymmetric kernel -> ~ 0.0)" )
print( f"     2nd-order sum = { tor.sum( CoeffsDef[ 2 ] ):.6f}  (zero DC response -> ~ 0.0)\n" )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3b. Visualize filters for different FilterOrders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  3b. Visualizing filter shapes (see Figure 1):" )
print( "     Larger FilterOrder = wider mollifier = more smoothing but poorer temporal localization." )
print( "     Smaller FilterOrder = narrower = sharper derivatives but noisier.\n" )

FilterOrders: list[ int ] = [ 11, 31, 61 ]
with plt.style.context( 'dark_background' ):
  FigFilt, AxsFilt = plt.subplots( 1, len( FilterOrders ), figsize = ( 15, 4 ) )
  for i, fo in enumerate( FilterOrders ):
    C = DiffedGaussianMollifier( FilterOrder = fo, nDerivatives = 2, std = Std, Plot = False )
    for d in range( len( C ) ): AxsFilt[ i ].plot( C[ d ].cpu().numpy(), label = f"d{ d } Gaussian" )
    AxsFilt[ i ].set_title( f"FilterOrder = { fo }" )
    AxsFilt[ i ].legend( fontsize = 8 )
    AxsFilt[ i ].grid( True, alpha = 0.3 )
  FigFilt.tight_layout()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3c. Smooth peak derivative properties ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  3c. Smooth peak derivative properties:" )
print( "     * Derivatives have [derivative_order]+1 peaks alternating in sign." )
print( "     * Even-order derivatives have a central peak; odd-order are zero-crossed at center." )
print( f"     * Larger bell (FilterOrder={ FilterOrders[ -1 ] }) = flatter derivatives even at same amplitude.\n" )

# ~~~~~~~~~~~~~~~~~~~~~~~~ 3d. Similarity to finite differences ~~~~~~~~~~~~~~~~~~~~~~~~
print( "  3d. Similarity to finite differences:" )
print( "     Low FIR order differentiated Gaussians resemble mirrored classical finite difference stencils." )

for fo_small in [ 5, 7 ]:
  Cs = DiffedGaussianMollifier( FilterOrder = fo_small, nDerivatives = 2, std = 0.15, Plot = False )
  mid = fo_small // 2
  print( f"     FilterOrder={ fo_small }:" )
  print( f"       d^1 center taps ~ { np.round( Cs[ 1 ].cpu().numpy()[ mid - 2 : mid + 3 ], 3 ) }" )
  print( f"       d^2 center taps ~ { np.round( Cs[ 2 ].cpu().numpy()[ mid - 2 : mid + 3 ], 3 ) }" )
print( "     Note: Unlike classical stencils, the 3rd-order mollifier derivative does NOT vanish at center.\n" )

# ---------------------------------------------------- 4. Simulate the Coupled Spring-Mass-Damper
print( "#" * 80 )
print( "  Section 4: Simulating the Coupled Spring-Mass-Damper System" )
print( "#" * 80 + "\n" )

# Equations of motion:
#   m1*x1'' + c1*x1' + k1*x1 + k2*(x1 - x2) + c2*(x1' - x2') = u(t)
#   m2*x2'' + c2*(x2' - x1') + k2*(x2 - x1) = 0
#
# Rearranged:
#   x1'' = (-(c1+c2)*x1' + c2*x2' - (k1+k2)*x1 + k2*x2 + u) / m1
#   x2'' = (c2*x1' - c2*x2' + k2*x1 - k2*x2) / m2

# Chirp input: frequency sweep from 0.1 to 5 Hz to excite both vibration modes
t_vals: np.ndarray = np.linspace( 0, T, p )
f0, f1 = 0.1, 5.0
u_vals: np.ndarray = np.sin( 2 * np.pi * ( f0 * t_vals + ( f1 - f0 ) * t_vals**2 / ( 2 * T ) ) )

def MakeODERHS( t_vals: np.ndarray, u_vals: np.ndarray ):
  '''Returns the ODE RHS function with closed-over interpolation of the chirp input.'''
  def ode_rhs( t: float, state: np.ndarray ) -> list[ float ]:
    x1, v1, x2, v2 = state
    u_t: float = float( np.interp( t, t_vals, u_vals ) )
    a1: float = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_t ) / m1
    a2: float = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2
    return [ v1, a1, v2, a2 ]
  return ode_rhs

sol = spi.solve_ivp( MakeODERHS( t_vals, u_vals ), [ 0, T ], [ 0.0, 0.0, 0.0, 0.0 ],
                     t_eval = t_vals, method = 'RK45' )

x1: tor.Tensor = tor.tensor( sol.y[ 0 ], dtype = tor.float64 )
v1: tor.Tensor = tor.tensor( sol.y[ 1 ], dtype = tor.float64 ) # True (analytical) velocity
x2: tor.Tensor = tor.tensor( sol.y[ 2 ], dtype = tor.float64 )
v2: tor.Tensor = tor.tensor( sol.y[ 3 ], dtype = tor.float64 ) # True velocity
u_sig: tor.Tensor = tor.tensor( u_vals, dtype = tor.float64 )

# Analytical accelerations (from the ODE RHS -- these are the "ground truth" for the accelerations)
a1_true: tor.Tensor = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_sig ) / m1
a2_true: tor.Tensor = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2

print( f"  Simulated { p } samples at { Fs } Hz ({ T:.1f} s)" )
print( f"  Chirp input: { f0 } -> { f1 } Hz frequency sweep" )
print( f"  System modes: ~{ np.sqrt( ( k1 + k2 + k2 + np.sqrt( ( k1 + k2 - k2 )**2 + 4 * k2**2 ) ) / ( 2 ) ):.2f} rad/s and "
       f"~{ np.sqrt( ( k1 + k2 + k2 - np.sqrt( ( k1 + k2 - k2 )**2 + 4 * k2**2 ) ) / ( 2 ) ):.2f} rad/s\n" )

# Plot the raw simulation outputs
with plt.style.context( 'dark_background' ):
  FigSim, AxsSim = plt.subplots( 3, 1, figsize = ( 12, 8 ), sharex = True )
  Time: np.ndarray = np.arange( p ) * dt
  AxsSim[ 0 ].plot( Time, x1.cpu().numpy(), label = "x_1 (mass 1)" )
  AxsSim[ 0 ].plot( Time, x2.cpu().numpy(), label = "x_2 (mass 2)" )
  AxsSim[ 0 ].set_ylabel( "Position" ); AxsSim[ 0 ].legend(); AxsSim[ 0 ].grid( True, alpha = 0.3 )
  AxsSim[ 1 ].plot( Time, v1.cpu().numpy(), label = "v_1 (mass 1)" )
  AxsSim[ 1 ].plot( Time, v2.cpu().numpy(), label = "v_2 (mass 2)" )
  AxsSim[ 1 ].set_ylabel( "Velocity" ); AxsSim[ 1 ].legend(); AxsSim[ 1 ].grid( True, alpha = 0.3 )
  AxsSim[ 2 ].plot( Time, u_sig.cpu().numpy(), label = "u(t) -- chirp input" )
  AxsSim[ 2 ].set_ylabel( "Force" ); AxsSim[ 2 ].set_xlabel( "Time [s]" )
  AxsSim[ 2 ].legend(); AxsSim[ 2 ].grid( True, alpha = 0.3 )
  FigSim.tight_layout()

# ---------------------------------------------------- 5. ODE Discovery via DerivativeAugment
print( "#" * 80 )
print( "  Section 5: ODE Discovery via DerivativeAugment" )
print( "#" * 80 + "\n" )

# Build the measurement matrix: columns [x1, x2, u] -- only positions and force, NOT velocities
Data: tor.Tensor = tor.column_stack( [ x1, x2, u_sig ] )
RegNamesIn: list[ str ] = [ "x1", "x2", "u" ]

# Augment with mollifier-smoothed derivatives up to order 2
# Output: [x1, x2, u,  d^1x1, d^1x2, d^1u,  d^2x1, d^2x2, d^2u]
AugData, AugNames = SmoothDeriver( Data, RegNamesIn, nDerivatives = nDerivatives,
                                        FilterOrder = FilterOrder )
print( f"  Augmented matrix: { Data.shape[ 0 ] } samples x { AugData.shape[ 1 ] } columns" )
SafeNames: list[ str ] = [ _sanitize( str( n ) ) for n in AugNames ]
print( f"  Column layout: { SafeNames }\n" )
print( "  Column indices:" )
for idx, name in enumerate( SafeNames ): print( f"    [{ idx }] { name }" )
print( "" )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 5a. Fit the x_1'' equation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  " + "-" * 56 )
print( "    5a. Recovering the x_1'' (mass 1) equation" )
print( "  " + "-" * 56 )

# Target = d^2x1 (column 6)
# Regressors = x1(0), x2(1), u(2), d^1x1(3), d^1x2(4)
# True ODE: x1'' = -(k1+k2)/m1*x1 + k2/m1*x2 + 1/m1*u - (c1+c2)/m1*x1' + c2/m1*x2'
TrueCoeffs1: tor.Tensor = tor.tensor( [ -( k1 + k2 ) / m1, k2 / m1, 1.0 / m1,
                                        -( c1 + c2 ) / m1, c2 / m1 ], dtype = tor.float64 )
RegNames1: list[ str ] = [ "x1", "x2", "u", "d^1 x1", "d^1 x2" ]
TargetIdx1: int = 6 # d^2x1 column
RegIdx1: list[ int ] = [ 0, 1, 2, 3, 4 ]

RegMat1: tor.Tensor = AugData[ :, RegIdx1 ]
Target1: tor.Tensor = AugData[ :, TargetIdx1 ]
Result1 = tor.linalg.lstsq( RegMat1, Target1 )
Coeffs1: tor.Tensor = Result1.solution

print( f"\n  { 'Term':>12s}  { 'True':>10s}  { 'Recovered':>10s}  { 'Error':>10s}" )
print( f"  { '-' * 44 }" )
for i in range( len( RegNames1 ) ):
  err = ( Coeffs1[ i ] - TrueCoeffs1[ i ] ).item()
  print( f"  { RegNames1[ i ]:>12s}  { TrueCoeffs1[ i ].item():10.4f}  { Coeffs1[ i ].item():10.4f}  { err:+10.4f}" )

mse1: float = tor.mean( ( Target1 - RegMat1 @ Coeffs1 )**2 ).item()
print( f"\n  MSE (d^2x_1 prediction vs DerivativeAugment target): { mse1:.2e}" )
print( f"  Relative error: { 100 * mse1 / tor.var( Target1 ).item():.4f}%\n" )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 5b. Fit the x_2'' equation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  " + "-" * 56 )
print( "    5b. Recovering the x_2'' (mass 2) equation" )
print( "  " + "-" * 56 )

# Target = d^2x2 (column 7)
# Regressors = x1(0), x2(1), d^1x1(3), d^1x2(4)  -- no u term in mass 2's equation
# True ODE: x2'' = k2/m2*x1 - k2/m2*x2 + c2/m2*x1' - c2/m2*x2'
TrueCoeffs2: tor.Tensor = tor.tensor( [ k2 / m2, -k2 / m2, c2 / m2, -c2 / m2 ], dtype = tor.float64 )
RegNames2: list[ str ] = [ "x1", "x2", "d^1 x1", "d^1 x2" ]
TargetIdx2: int = 7 # d^2x2 column
RegIdx2: list[ int ] = [ 0, 1, 3, 4 ]

RegMat2: tor.Tensor = AugData[ :, RegIdx2 ]
Target2: tor.Tensor = AugData[ :, TargetIdx2 ]
Result2 = tor.linalg.lstsq( RegMat2, Target2 )
Coeffs2: tor.Tensor = Result2.solution

print( f"\n  { 'Term':>12s}  { 'True':>10s}  { 'Recovered':>10s}  { 'Error':>10s}" )
print( f"  { '-' * 44 }" )
for i in range( len( RegNames2 ) ):
  err = ( Coeffs2[ i ] - TrueCoeffs2[ i ] ).item()
  print( f"  { RegNames2[ i ]:>12s}  { TrueCoeffs2[ i ].item():10.4f}  { Coeffs2[ i ].item():10.4f}  { err:+10.4f}" )

mse2: float = tor.mean( ( Target2 - RegMat2 @ Coeffs2 )**2 ).item()
print( f"\n  MSE (d^2x_2 prediction vs DerivativeAugment target): { mse2:.2e}" )
print( f"  Relative error: { 100 * mse2 / tor.var( Target2 ).item():.4f}%\n" )

# ---------------------------------------------------- 6. Validate the Recovered Model
print( "#" * 80 )
print( "  Section 6: Validation -- Recovered Model vs. Ground Truth" )
print( "#" * 80 + "\n" )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 6a. Compare acceleration predictions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  6a. Acceleration prediction (using recovered coefficients on DerivativeAugment signals):" )

a1_pred: tor.Tensor = RegMat1 @ Coeffs1
a2_pred: tor.Tensor = RegMat2 @ Coeffs2

err_a1: float = tor.mean( ( a1_true - a1_pred )**2 ).item()
err_a2: float = tor.mean( ( a2_true - a2_pred )**2 ).item()
print( f"     MSE(x_1''): { err_a1:.4e}" )
print( f"     MSE(x_2''): { err_a2:.4e}\n" )

with plt.style.context( 'dark_background' ):
  FigVal, AxsVal = plt.subplots( 2, 1, figsize = ( 12, 6 ), sharex = True )
  AxsVal[ 0 ].plot( Time, a1_true.cpu().numpy(), label = "True x_1'' (analytical ODE)", alpha = 0.8 )
  AxsVal[ 0 ].plot( Time, a1_pred.cpu().numpy(), '--', label = "Recovered x_1'' (DerivativeAugment)", alpha = 0.8, linewidth = 1.5 )
  AxsVal[ 0 ].set_ylabel( "Acceleration" ); AxsVal[ 0 ].legend(); AxsVal[ 0 ].grid( True, alpha = 0.3 )
  AxsVal[ 1 ].plot( Time, a2_true.cpu().numpy(), label = "True x_2'' (analytical ODE)", alpha = 0.8 )
  AxsVal[ 1 ].plot( Time, a2_pred.cpu().numpy(), '--', label = "Recovered x_2'' (DerivativeAugment)", alpha = 0.8, linewidth = 1.5 )
  AxsVal[ 1 ].set_ylabel( "Acceleration" ); AxsVal[ 1 ].set_xlabel( "Time [s]" )
  AxsVal[ 1 ].legend(); AxsVal[ 1 ].grid( True, alpha = 0.3 )
  FigVal.tight_layout()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 6b. Simulate the recovered ODE forward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print( "  6b. Full trajectory simulation with recovered ODE (integrating the discovered equation):" )
print( "      Initial condition = same as ground truth, same chirp input." )

def MakeRecoveredRHS( t_vals: np.ndarray, u_vals: np.ndarray, Coeffs1: tor.Tensor, Coeffs2: tor.Tensor ):
  '''ODE RHS using the coefficients recovered by DerivativeAugment regression.'''
  c1_np = Coeffs1.detach().cpu().numpy()
  c2_np = Coeffs2.detach().cpu().numpy()
  def rhs( t: float, state: np.ndarray ) -> list[ float ]:
    x1, v1, x2, v2 = state
    u_t = float( np.interp( t, t_vals, u_vals ) )
    a1 = c1_np[ 0 ] * x1 + c1_np[ 1 ] * x2 + c1_np[ 2 ] * u_t + c1_np[ 3 ] * v1 + c1_np[ 4 ] * v2
    a2 = c2_np[ 0 ] * x1 + c2_np[ 1 ] * x2 + c2_np[ 2 ] * v1 + c2_np[ 3 ] * v2
    return [ v1, float( a1 ), v2, float( a2 ) ]
  return rhs

solRec = spi.solve_ivp( MakeRecoveredRHS( t_vals, u_vals, Coeffs1, Coeffs2 ),
                        [ 0, T ], sol.y[ :, 0 ], t_eval = t_vals, method = 'RK45' )

x1_rec: np.ndarray = solRec.y[ 0 ]
x2_rec: np.ndarray = solRec.y[ 2 ]

err_traj1: float = np.mean( ( sol.y[ 0 ] - x1_rec )**2 )
err_traj2: float = np.mean( ( sol.y[ 2 ] - x2_rec )**2 )
print( f"     MSE(x_1 trajectory): { err_traj1:.4e}" )
print( f"     MSE(x_2 trajectory): { err_traj2:.4e}\n" )

with plt.style.context( 'dark_background' ):
  FigTraj, AxsTraj = plt.subplots( 2, 1, figsize = ( 12, 6 ), sharex = True )
  AxsTraj[ 0 ].plot( Time, sol.y[ 0 ], label = "True x_1" )
  AxsTraj[ 0 ].plot( Time, x1_rec, '--', label = "Recovered x_1 (integrated)", linewidth = 1.5 )
  AxsTraj[ 0 ].set_ylabel( "Position" ); AxsTraj[ 0 ].legend(); AxsTraj[ 0 ].grid( True, alpha = 0.3 )
  AxsTraj[ 1 ].plot( Time, sol.y[ 2 ], label = "True x_2" )
  AxsTraj[ 1 ].plot( Time, x2_rec, '--', label = "Recovered x_2 (integrated)", linewidth = 1.5 )
  AxsTraj[ 1 ].set_ylabel( "Position" ); AxsTraj[ 1 ].set_xlabel( "Time [s]" )
  AxsTraj[ 1 ].legend(); AxsTraj[ 1 ].grid( True, alpha = 0.3 )
  FigTraj.tight_layout()

# ---------------------------------------------------- 7. What Happens Without DerivativeAugment?
print( "#" * 80 )
print( "  Section 7: Without DerivativeAugment -- The Fit Fails" )
print( "#" * 80 + "\n" )
print( "  If we only use positions and the input (no velocity/acceleration columns), the regression" )
print( "  cannot capture the ODE structure since it's missing the damping (velocity-dependent) terms.\n" )

# Try fitting x1'' using x1, x2, u only (no d^1 columns) -- should be a terrible fit
RegMat_noDeriv: tor.Tensor = AugData[ :, [ 0, 1, 2 ] ]
Result_noDeriv = tor.linalg.lstsq( RegMat_noDeriv, Target1 )
Coeffs_noDeriv: tor.Tensor = Result_noDeriv.solution
a1_pred_noDeriv: tor.Tensor = RegMat_noDeriv @ Coeffs_noDeriv

mse_with: float = tor.mean( ( a1_pred - a1_true )**2 ).item()
mse_without: float = tor.mean( ( a1_pred_noDeriv - a1_true )**2 ).item()

print( f"  x_1'' prediction MSE with    DerivativeAugment: { mse_with:.4e}" )
print( f"  x_1'' prediction MSE without DerivativeAugment: { mse_without:.4e}" )
print( f"  Ratio: { mse_without / mse_with:.0f}x worse without derivatives!\n" )

with plt.style.context( 'dark_background' ):
  FigNoDeriv, AxNoDeriv = plt.subplots( figsize = ( 12, 4 ) )
  AxNoDeriv.plot( Time, a1_true.cpu().numpy(), label = "True x_1''", alpha = 0.8 )
  AxNoDeriv.plot( Time, a1_pred_noDeriv.cpu().numpy(), '--', label = "x_1'' without derivatives (poor)", alpha = 0.8 )
  AxNoDeriv.set_ylabel( "Acceleration" ); AxNoDeriv.set_xlabel( "Time [s]" )
  AxNoDeriv.legend(); AxNoDeriv.grid( True, alpha = 0.3 )
  AxNoDeriv.set_title( "Without DerivativeAugment -- the regression cannot capture the ODE" )
  FigNoDeriv.tight_layout()

# ---------------------------------------------------- 8. Effect of FilterOrder on Derivative Quality
print( "#" * 80 )
print( "  Section 8: Bonus -- How FilterOrder Affects Derivative Quality" )
print( "#" * 80 + "\n" )

FilterOrderCandidates: list[ int ] = [ 7, 15, 31, 61, 101 ]

print( f"  Comparing FilterOrders { FilterOrderCandidates }:" )
print( f"  { 'FilterOrder':>14s}  { 'MSE(x_1'')':>12s}  { 'MSE(x_2'')':>12s}  { 'Notes':>30s}" )
print( f"  { '-' * 70 }" )

for fo in FilterOrderCandidates:
  AD_fo, _ = SmoothDeriver( Data, RegNamesIn, nDerivatives = nDerivatives, FilterOrder = fo )
  a1_pred_fo = AD_fo[ :, RegIdx1 ] @ tor.linalg.lstsq( AD_fo[ :, RegIdx1 ], AD_fo[ :, TargetIdx1 ] ).solution
  a2_pred_fo = AD_fo[ :, RegIdx2 ] @ tor.linalg.lstsq( AD_fo[ :, RegIdx2 ], AD_fo[ :, TargetIdx2 ] ).solution
  m1_fo = tor.mean( ( a1_true - a1_pred_fo )**2 ).item()
  m2_fo = tor.mean( ( a2_true - a2_pred_fo )**2 ).item()
  note = "noisy/edge-heavy" if fo < 11 else "good trade-off" if fo < 51 else "heavy smoothing"
  print( f"  { fo:>14d}  { m1_fo:>12.4e}  { m2_fo:>12.4e}  { note:>30s}" )

print( "\n  Rule of thumb: FilterOrder ~ 31 is a good starting point for most applications." )
print( "  Increase for very noisy data (more smoothing), decrease for fast transients (less smearing).\n" )

# ---------------------------------------------------- 9. Show Everything
plt.show()

print( "\n  Done! The mollifier-based DerivativeAugment successfully recovered the coupled ODE." )
print( "  Key takeaway: smooth the signal (LPF) before differentiating (HPF) to stabilize derivative estimation." )
print( "  This is what DiffedMollifiersCTor does -- and DerivativeAugment puts it to work.\n" )
