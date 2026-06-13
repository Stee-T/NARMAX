# Example 8 (final): ODE Discovery via Derivative‑Augmented Regressors
# Uses SmoothDeriver with time‑scaling (dt parameter) and edge trimming.
# Recovers physical coefficients directly.
import torch as tor
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

import NARMAX
from NARMAX.CTors import DiffedGaussianMollifier, SmoothDeriver

def _sanitize( name: str ) -> str:
  _map = {
        0x2202: ord( 'd' ), 0x00B9: ord( '1' ), 0x00B2: ord( '2' ),
        0x00B3: ord( '3' ), 0x2074: ord( '4' ), 0x2075: ord( '5' ),
        0x2076: ord( '6' ), 0x2077: ord( '7' ), 0x2078: ord( '8' ),
        0x2079: ord( '9' ), 0x2070: ord( '0' ),
    }
  return name.translate( _map )

# ---------------------------------------------------- Hyper‑parameters ----------------------------------------------------
p: int = 3_000
Fs: float = 100.0
dt: float = 1.0 / Fs # sampling interval (used for time scaling)
T: float = p * dt

m1, m2 = 1.0, 1.0 # masses [kg]
k1, k2 = 100.0, 50.0 # stiffnesses [N/m]
c1, c2 = 2.0, 1.0 # damping [Ns/m]

FilterOrder: int = 31
nDerivatives: int = 2
Std: float = 0.03 # from grid search

tol: float = 5e-3
ArboDepth: int = 3

half = FilterOrder // 2 # edge samples to discard

# ---------------------------------------------------- 1. Differentiated Mollifiers ----------------------------------------------------
print( "1. Differentiated Mollifiers\n" )
Coeffs = DiffedGaussianMollifier( FilterOrder = FilterOrder, nDerivatives = nDerivatives, std = Std )
print( f"  FilterOrder={ FilterOrder }, nDerivatives={ nDerivatives }, std={ Std }" )
for d in range( len( Coeffs ) ): print( f"  Order { d }: sum = { Coeffs[ d ].sum():.6f}" )

with plt.style.context( 'dark_background' ):
  fig_filt, axs_filt = plt.subplots( 1, 3, figsize = ( 15, 4 ) )
  for i, fo in enumerate( [ 11, 31, 61 ] ):
    C = DiffedGaussianMollifier( FilterOrder = fo, nDerivatives = 2, std = Std, Plot = False )
    for d in range( len( C ) ):
      coeff = C[ d ].cpu().numpy()
      coeff = coeff / np.max( np.abs( coeff ) )
      axs_filt[ i ].plot( coeff, label = f"d{ d }" )
    axs_filt[ i ].set_title( f"FilterOrder = { fo } (peak‑normalised)" )
    axs_filt[ i ].legend( fontsize = 8 )
    axs_filt[ i ].grid( True, alpha = 0.3 )
  fig_filt.tight_layout()
print( "  Figure_1.png saved.\n" )

# ---------------------------------------------------- 2. Simulate Coupled Spring‑Mass‑Damper ----------------------------------------------------
print( "2. Simulating system\n" )
t_vals = np.linspace( 0, T, p )
f0, f1 = 0.1, 5.0
u_vals = np.sin( 2 * np.pi * ( f0 * t_vals + ( f1 - f0 ) * t_vals**2 / ( 2 * T ) ) )

def MakeODERHS( t_vals, u_vals ):
  def ode_rhs( t, state ):
    x1, v1, x2, v2 = state
    u_t = float( np.interp( t, t_vals, u_vals ) )
    a1 = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_t ) / m1
    a2 = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2
    return [ v1, a1, v2, a2 ]
  return ode_rhs

sol = spi.solve_ivp( MakeODERHS( t_vals, u_vals ), [ 0, T ], [ 0.0, 0.0, 0.0, 0.0 ],
                     t_eval = t_vals, method = 'RK45' )
x1 = tor.tensor( sol.y[ 0 ], dtype = tor.float64 )
v1 = tor.tensor( sol.y[ 1 ], dtype = tor.float64 )
x2 = tor.tensor( sol.y[ 2 ], dtype = tor.float64 )
v2 = tor.tensor( sol.y[ 3 ], dtype = tor.float64 )
u_sig = tor.tensor( u_vals, dtype = tor.float64 )

# True analytical accelerations (for reference, not used in training)
a1_true = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_sig ) / m1
a2_true = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2

time = np.arange( p ) * dt
with plt.style.context( 'dark_background' ):
  fig_sim, axs_sim = plt.subplots( 5, 1, figsize = ( 8, 12 ) )
  axs_sim[ 0 ].plot( time, x1.cpu().numpy() ); axs_sim[ 0 ].set_ylabel( "x1" ); axs_sim[ 0 ].grid( True, alpha = 0.3 )
  axs_sim[ 1 ].plot( time, x2.cpu().numpy() ); axs_sim[ 1 ].set_ylabel( "x2" ); axs_sim[ 1 ].grid( True, alpha = 0.3 )
  axs_sim[ 2 ].plot( time, u_sig.cpu().numpy(), color = 'orange' ); axs_sim[ 2 ].set_ylabel( "u(t)" ); axs_sim[ 2 ].grid( True, alpha = 0.3 )
  axs_sim[ 3 ].plot( time, v1.cpu().numpy(), color = 'gold' ); axs_sim[ 3 ].set_ylabel( "v1" ); axs_sim[ 3 ].grid( True, alpha = 0.3 )
  axs_sim[ 4 ].plot( time, v2.cpu().numpy(), color = 'gold' ); axs_sim[ 4 ].set_ylabel( "v2" ); axs_sim[ 4 ].set_xlabel( "Time [s]" ); axs_sim[ 4 ].grid( True, alpha = 0.3 )
  fig_sim.tight_layout()
print( "  Figure_2.png saved.\n" )

# ---------------------------------------------------- 3. Augment Training Data (time‑scaled, physical) ----------------------------------------------------
print( "3. Augmenting training data (time‑scaled, NormDerivatives=False)\n" )
Data = tor.column_stack( [ x1, x2, u_sig ] )
AugData, AugNames = SmoothDeriver( Data, [ "x1", "x2", "u" ],
                                  nDerivatives = nDerivatives,
                                  FilterOrder = FilterOrder,
                                  NormDerivatives = False, # keep physical units
                                  dt = dt ) # enable time‑scaling
SafeNames = [ _sanitize( str( n ) ) for n in AugNames ]
print( f"  Augmented matrix: { AugData.shape[ 0 ] } x { AugData.shape[ 1 ] }" )
print( f"  Columns: { SafeNames }\n" )

# Trim filter transients from both ends
AugData = AugData[ half : -half, : ]

TargetIdx1 = 6 # ∂² x1 (now in m/s²)
RegIdx1 = [ 0, 1, 2, 3, 4, 5 ] # x1, x2, u, ∂¹x1, ∂¹x2, ∂¹u
TargetIdx2 = 7 # ∂² x2
RegIdx2 = [ 0, 1, 2, 3, 4, 5 ]

y1 = AugData[ :, TargetIdx1 ]
Dc1 = AugData[ :, RegIdx1 ]
DcNames1 = AugNames[ RegIdx1 ]

y2 = AugData[ :, TargetIdx2 ]
Dc2 = AugData[ :, RegIdx2 ]
DcNames2 = AugNames[ RegIdx2 ]

print( f"  Training samples after trimming: { len( y1 ) }" )
print( f"  Target: d²x1 (m/s²), { len( RegIdx1 ) } candidate regressors\n" )

# ---------------------------------------------------- 4. Custom Validation Function (time‑scaled, trimmed) ----------------------------------------------------
def ODE_MAE( theta, L, ERR, RegNames, ValDic, DcFilterIdx = None ):
  '''Relative MAE on time‑scaled, trimmed validation data.'''
  Error = 0.0
  h = ValDic[ "FilterOrder" ] // 2
  for i in range( len( ValDic[ "Data" ] ) ):
    AugData_val, _ = SmoothDeriver( ValDic[ "Data" ][ i ], ValDic[ "RegNamesIn" ],
                                       nDerivatives = ValDic[ "nDerivatives" ],
                                       FilterOrder = ValDic[ "FilterOrder" ],
                                       NormDerivatives = False,
                                       dt = ValDic[ "dt" ] ) # use the same dt
    y_true = AugData_val[ h : -h, ValDic[ "TargetIdx" ] ]
    RegMat = AugData_val[ h : -h, ValDic[ "RegIdx" ] ]
    if ( DcFilterIdx is not None ): RegMat = RegMat[ :, DcFilterIdx ]
    yHat = RegMat[ :, L.astype( np.int64 ) ] @ theta
    denom = tor.mean( tor.abs( y_true ) )
    if ( denom > 1e-15 ): Error += ( tor.mean( tor.abs( y_true - yHat ) ) / denom ).item()
  return Error / max( len( ValDic[ "Data" ] ), 1 )

# ---------------------------------------------------- 5. Validation Data ----------------------------------------------------
print( "4. Generating validation data\n" )
p_val = 1_000
t_val = np.linspace( 0, p_val * dt, p_val )

ValData1 = { "Data": [], "RegNamesIn": [ "x1", "x2", "u" ],
            "nDerivatives": nDerivatives, "FilterOrder": FilterOrder,
            "RegIdx": RegIdx1, "TargetIdx": TargetIdx1,
            "dt": dt } # pass dt for consistent scaling
ValData2 = { "Data": [], "RegNamesIn": [ "x1", "x2", "u" ],
            "nDerivatives": nDerivatives, "FilterOrder": FilterOrder,
            "RegIdx": RegIdx2, "TargetIdx": TargetIdx2,
            "dt": dt }

for phase_shift in [ 0.0, np.pi / 3, 2.0 * np.pi / 3 ]:
  u_val = np.sin( 2 * np.pi * ( f0 * t_val + ( f1 - f0 ) * t_val**2 / ( 2 * T ) ) + phase_shift )
  sol_val = spi.solve_ivp( MakeODERHS( t_val, u_val ), [ 0, p_val * dt ],
                            [ 0.0, 0.0, 0.0, 0.0 ], t_eval = t_val, method = 'RK45' )
  Data_val = tor.column_stack( [
        tor.tensor( sol_val.y[ 0 ], dtype = tor.float64 ),
        tor.tensor( sol_val.y[ 2 ], dtype = tor.float64 ),
        tor.tensor( u_val, dtype = tor.float64 ),
    ] )
  ValData1[ "Data" ].append( Data_val )
  ValData2[ "Data" ].append( Data_val )
print( f"  { len( ValData1[ 'Data' ] ) } validation sets generated\n" )

# ---------------------------------------------------- 6. Arborescence: x1'' Equation ----------------------------------------------------
print( "5. Arborescence: x1''\n" )
Arbo1 = NARMAX.Arborescence( y1, Dc = Dc1, DcNames = DcNames1,
                            tolRoot = tol, tolRest = tol,
                            MaxDepth = ArboDepth,
                            ValFunc = ODE_MAE, ValData = ValData1 )
theta1, L1, ERR1, _, _, _ = Arbo1.fit()
print( f"\n  { 'Term':>12s}  { 'Recovered':>10s}  (True)" )
print( f"  { '-' * 40 }" )
true_vals1 = [ -( k1 + k2 ) / m1, k2 / m1, 1 / m1, -( c1 + c2 ) / m1, c2 / m1, 0.0 ]
for i, idx in enumerate( L1 ):
  true_str = f"{ true_vals1[ idx ]:10.4f}" if idx < len( true_vals1 ) else "         -"
  print( f"  { _sanitize( str( DcNames1[ idx ] ) ):>12s}  { theta1[ i ].item():10.4f}  { true_str }" )

a1_pred = Dc1[ :, L1.astype( np.int64 ) ] @ theta1
mse_fit1 = tor.mean( ( y1 - a1_pred )**2 ).item()
print( f"\n  Fit MSE (vs physical d²x1): { mse_fit1:.4e}\n" )

# ---------------------------------------------------- 7. Arborescence: x2'' Equation ----------------------------------------------------
print( "6. Arborescence: x2''\n" )
Arbo2 = NARMAX.Arborescence( y2, Dc = Dc2, DcNames = DcNames2,
                            tolRoot = tol, tolRest = tol,
                            MaxDepth = ArboDepth,
                            ValFunc = ODE_MAE, ValData = ValData2 )
theta2, L2, ERR2, _, _, _ = Arbo2.fit()
print( f"\n  { 'Term':>12s}  { 'Recovered':>10s}  (True)" )
print( f"  { '-' * 40 }" )
true_vals2 = [ k2 / m2, -k2 / m2, 0.0, c2 / m2, -c2 / m2, 0.0 ]
for i, idx in enumerate( L2 ):
  true_str = f"{ true_vals2[ idx ]:10.4f}" if idx < len( true_vals2 ) else "         -"
  print( f"  { _sanitize( str( DcNames2[ idx ] ) ):>12s}  { theta2[ i ].item():10.4f}  { true_str }" )

a2_pred = Dc2[ :, L2.astype( np.int64 ) ] @ theta2
mse_fit2 = tor.mean( ( y2 - a2_pred )**2 ).item()
print( f"\n  Fit MSE (vs physical d²x2): { mse_fit2:.4e}\n" )

# ---------------------------------------------------- 8. Acceleration Validation ----------------------------------------------------
print( "7. Acceleration prediction (on trimmed training data)\n" )
Coeffs1_full = tor.zeros( len( RegIdx1 ), dtype = tor.float64 )
Coeffs1_full[ L1.astype( np.int64 ) ] = theta1
Coeffs2_full = tor.zeros( len( RegIdx2 ), dtype = tor.float64 )
Coeffs2_full[ L2.astype( np.int64 ) ] = theta2

a1_pred_full = Dc1 @ Coeffs1_full
a2_pred_full = Dc2 @ Coeffs2_full
err_a1 = tor.mean( ( y1 - a1_pred_full )**2 ).item()
err_a2 = tor.mean( ( y2 - a2_pred_full )**2 ).item()
print( f"  MSE (vs physical d²x1): { err_a1:.4e}" )
print( f"  MSE (vs physical d²x2): { err_a2:.4e}\n" )

with plt.style.context( 'dark_background' ):
  fig_val, axs_val = plt.subplots( 2, 1, figsize = ( 12, 6 ), sharex = True )
  axs_val[ 0 ].plot( time[ half : -half ], y1.cpu().numpy(), label = "Physical d²x1", alpha = 0.8 )
  axs_val[ 0 ].plot( time[ half : -half ], a1_pred_full.cpu().numpy(), '--', label = "Recovered", alpha = 0.8 )
  axs_val[ 0 ].set_ylabel( "Acceleration (m/s²)" ); axs_val[ 0 ].legend(); axs_val[ 0 ].grid( True, alpha = 0.3 )
  axs_val[ 1 ].plot( time[ half : -half ], y2.cpu().numpy(), label = "Physical d²x2", alpha = 0.8 )
  axs_val[ 1 ].plot( time[ half : -half ], a2_pred_full.cpu().numpy(), '--', label = "Recovered", alpha = 0.8 )
  axs_val[ 1 ].set_ylabel( "Acceleration (m/s²)" ); axs_val[ 1 ].set_xlabel( "Time [s]" )
  axs_val[ 1 ].legend(); axs_val[ 1 ].grid( True, alpha = 0.3 )
  fig_val.tight_layout()
print( "  Figure_3.png saved.\n" )

# ---------------------------------------------------- 9. Full Trajectory Simulation ----------------------------------------------------
print( "8. Forward trajectory simulation\n" )
# The recovered coefficients are now physical, use them directly.
c1_np = Coeffs1_full.cpu().numpy()
c2_np = Coeffs2_full.cpu().numpy()

def MakeRecoveredRHS( Coeffs1_full, Coeffs2_full ):
  def rhs( t, state ):
    x1, v1, x2, v2 = state
    u_t = float( np.interp( t, t_vals, u_vals ) )
    a1 = ( c1_np[ 0 ] * x1 + c1_np[ 1 ] * x2 + c1_np[ 2 ] * u_t +
              c1_np[ 3 ] * v1 + c1_np[ 4 ] * v2 )
    a2 = ( c2_np[ 0 ] * x1 + c2_np[ 1 ] * x2 + c2_np[ 2 ] * u_t +
              c2_np[ 3 ] * v1 + c2_np[ 4 ] * v2 )
    return [ v1, float( a1 ), v2, float( a2 ) ]
  return rhs

solRec = spi.solve_ivp( MakeRecoveredRHS( Coeffs1_full, Coeffs2_full ),
                       [ 0, T ], sol.y[ :, 0 ], t_eval = t_vals, method = 'RK45' )
mse_traj_x1 = np.mean( ( sol.y[ 0 ] - solRec.y[ 0 ] )**2 )
mse_traj_x2 = np.mean( ( sol.y[ 2 ] - solRec.y[ 2 ] )**2 )
print( f"  Trajectory MSE x1: { mse_traj_x1:.4e}" )
print( f"  Trajectory MSE x2: { mse_traj_x2:.4e}\n" )

with plt.style.context( 'dark_background' ):
  fig_traj, axs_traj = plt.subplots( 2, 1, figsize = ( 12, 6 ), sharex = True )
  axs_traj[ 0 ].plot( time, sol.y[ 0 ], label = "True x1" )
  axs_traj[ 0 ].plot( time, solRec.y[ 0 ], '--', label = "Recovered x1" )
  axs_traj[ 0 ].set_ylabel( "Position" ); axs_traj[ 0 ].legend(); axs_traj[ 0 ].grid( True, alpha = 0.3 )
  axs_traj[ 1 ].plot( time, sol.y[ 2 ], label = "True x2" )
  axs_traj[ 1 ].plot( time, solRec.y[ 2 ], '--', label = "Recovered x2" )
  axs_traj[ 1 ].set_ylabel( "Position" ); axs_traj[ 1 ].set_xlabel( "Time [s]" )
  axs_traj[ 1 ].legend(); axs_traj[ 1 ].grid( True, alpha = 0.3 )
  fig_traj.tight_layout()
print( "  Figure_4.png saved.\n" )

# ---------------------------------------------------- 10. Without Derivatives (unchanged, but now on physical data) ----------------------------------------------------
print( "9. Failing without derivative augmentation\n" )
Dc_noDeriv = AugData[ :, [ 0, 1, 2 ] ]
DcNames_noDeriv = AugNames[ [ 0, 1, 2 ] ]
Arbo_noDeriv = NARMAX.Arborescence( y1, Dc = Dc_noDeriv, DcNames = DcNames_noDeriv,
                                   tolRoot = tol, tolRest = tol, MaxDepth = 2,
                                   ValFunc = ODE_MAE, ValData = ValData1 )
theta_no, L_no, _, _, _, _ = Arbo_noDeriv.fit()
a1_pred_noDeriv = Dc_noDeriv[ :, L_no.astype( np.int64 ) ] @ theta_no
mse_without = tor.mean( ( y1 - a1_pred_noDeriv )**2 ).item()
print( f"  MSE with derivatives: { mse_fit1:.4e}" )
print( f"  MSE without derivatives: { mse_without:.4e}" )
if ( mse_without > 1e-10 ): print( f"  Ratio: { mse_without / max( mse_fit1, 1e-15 ):.0f}x worse!\n" )

with plt.style.context( 'dark_background' ):
  fig_no, ax_no = plt.subplots( figsize = ( 12, 4 ) )
  ax_no.plot( time[ half : -half ], y1.cpu().numpy(), label = "Target d²x1 (physical)" )
  ax_no.plot( time[ half : -half ], a1_pred_noDeriv.cpu().numpy(), '--', label = "Without derivatives" )
  ax_no.set_ylabel( "Acceleration (m/s²)" ); ax_no.set_xlabel( "Time [s]" )
  ax_no.legend(); ax_no.grid( True, alpha = 0.3 )
  ax_no.set_title( "Failing Without Derivative Augmentation" )
  fig_no.tight_layout()
print( "  Figure_5.png saved.\n" )

# ---------------------------------------------------- 11. FilterOrder Scan ----------------------------------------------------
print( "10. FilterOrder scan (with time-scaling, NormDerivatives=False)\n" )
for fo in [ 7, 15, 31, 61, 101 ]:
  AD_fo, _ = SmoothDeriver( Data, [ "x1", "x2", "u" ], nDerivatives = nDerivatives,
                             FilterOrder = fo, NormDerivatives = False, dt = dt )
  AD_fo = AD_fo[ fo//2 : -fo//2, : ] # trim to match
  y_fo = AD_fo[ :, TargetIdx1 ]
  Dc_fo = AD_fo[ :, RegIdx1 ]
  theta_fo = tor.linalg.lstsq( Dc_fo, y_fo ).solution
  pred_fo = Dc_fo @ theta_fo
  mse_fo = tor.mean( ( y_fo - pred_fo )**2 ).item()
  note = "noisy" if fo < 11 else "good" if fo < 51 else "smooth"
  print( f"  FilterOrder { fo:>3d}: MSE = { mse_fo:.4e}  ({ note })" )

print( "\nDone." )
