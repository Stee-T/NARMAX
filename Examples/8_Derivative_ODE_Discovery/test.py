import torch, numpy as np, scipy.integrate as spi
from NARMAX.CTors import SmoothDeriver, DiffedGaussianMollifier

# ------------------------------------------------------------
# 1. Generate training data (same as tutorial)
# ------------------------------------------------------------
p = 3000; Fs = 100.0; dt = 1.0 / Fs; T = p * dt
m1, m2 = 1.0, 1.0; k1, k2 = 100.0, 50.0; c1, c2 = 2.0, 1.0
t_vals = np.linspace( 0, T, p )
f0, f1 = 0.1, 5.0
u_vals = np.sin( 2 * np.pi * ( f0 * t_vals + ( f1 - f0 ) * t_vals**2 / ( 2 * T ) ) )

def true_ode( t, state ):
  x1, v1, x2, v2 = state
  u_t = float( np.interp( t, t_vals, u_vals ) )
  a1 = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_t ) / m1
  a2 = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2
  return [ v1, a1, v2, a2 ]

sol = spi.solve_ivp( true_ode, [ 0, T ], [ 0., 0., 0., 0. ], t_eval = t_vals, method = 'RK45' )
x1 = torch.tensor( sol.y[ 0 ], dtype = torch.float64 )
x2 = torch.tensor( sol.y[ 2 ], dtype = torch.float64 )
u = torch.tensor( u_vals, dtype = torch.float64 )
Data = torch.column_stack( [ x1, x2, u ] )
RegNames = [ "x1", "x2", "u" ]

FilterOrder = 31; nDeriv = 2; std = 0.03
half = FilterOrder//2
T_f = ( FilterOrder - 1 ) * dt # time span of the kernel

# ------------------------------------------------------------
# 2. Get mollifier kernels
# ------------------------------------------------------------
kernels = DiffedGaussianMollifier( FilterOrder, nDeriv, std )
kernel0 = kernels[ 0 ].to( torch.float64 ) # Gaussian
kernel1 = kernels[ 1 ].to( torch.float64 ) # d/dX
kernel2 = kernels[ 2 ].to( torch.float64 ) # d²/dX²

# ------------------------------------------------------------
# 3. Approach A: SmoothDeriver as‑is (no time scaling)
# ------------------------------------------------------------
AugA, _ = SmoothDeriver( Data, RegNames, nDerivatives = nDeriv,
                        FilterOrder = FilterOrder, std = std,
                        NormDerivatives = False )
AugA = AugA[ half : -half, : ] # trim edges
# columns: [x1_raw, x2_raw, u_raw, v1_x, v2_x, vu_x, a1_x, a2_x, au_x]
yA = AugA[ :, 6 ] # a1_x (in X‑units)
DcA = AugA[ :, [ 0, 1, 2, 3, 4, 5 ] ]
thetaA = torch.linalg.lstsq( DcA, yA ).solution

# ------------------------------------------------------------
# 4. Approach B: SmoothDeriver + manual time‑scaling of derivatives
# ------------------------------------------------------------
# Use the same AugA, but scale velocity columns by 1/T_f, acceleration by 1/T_f^2
AugB = AugA.clone()
# velocity columns are indices 3,4,5; acceleration 6,7,8
AugB[ :, 3 : 6 ] /= T_f
AugB[ :, 6 : 9 ] /= ( T_f**2 )
# Now the target is in m/s², velocities in m/s
yB = AugB[ :, 6 ]
DcB = AugB[ :, [ 0, 1, 2, 3, 4, 5 ] ]
thetaB = torch.linalg.lstsq( DcB, yB ).solution

# ------------------------------------------------------------
# 5. Approach C: NormDerivatives=True (as in current tutorial)
# ------------------------------------------------------------
AugC, _ = SmoothDeriver( Data, RegNames, nDerivatives = nDeriv,
                        FilterOrder = FilterOrder, std = std,
                        NormDerivatives = True )
AugC = AugC[ half : -half, : ]
yC = AugC[ :, 6 ] # normalised d²x
DcC = AugC[ :, [ 0, 1, 2, 3, 4, 5 ] ]
thetaC = torch.linalg.lstsq( DcC, yC ).solution

# ------------------------------------------------------------
# 6. Validation data (different phase)
# ------------------------------------------------------------
p_val = 1000; t_val = np.linspace( 0, p_val * dt, p_val )
u_val = np.sin( 2 * np.pi * ( f0 * t_val + ( f1 - f0 ) * t_val**2 / ( 2 * T ) ) + np.pi / 3 )
sol_val = spi.solve_ivp( true_ode, [ 0, p_val * dt ], [ 0., 0., 0., 0. ],
                        t_eval = t_val, method = 'RK45' )
Data_val = torch.column_stack( [ torch.tensor( sol_val.y[ 0 ], dtype = torch.float64 ),
                               torch.tensor( sol_val.y[ 2 ], dtype = torch.float64 ),
                               torch.tensor( u_val, dtype = torch.float64 ) ] )

# Build validation matrices for each approach
Aug_valA, _ = SmoothDeriver( Data_val, RegNames, nDerivatives = nDeriv,
                            FilterOrder = FilterOrder, std = std,
                            NormDerivatives = False )
Aug_valA = Aug_valA[ half : -half, : ]
y_valA = Aug_valA[ :, 6 ]
Dc_valA = Aug_valA[ :, [ 0, 1, 2, 3, 4, 5 ] ]

Aug_valB = Aug_valA.clone()
Aug_valB[ :, 3 : 6 ] /= T_f
Aug_valB[ :, 6 : 9 ] /= T_f**2
y_valB = Aug_valB[ :, 6 ]
Dc_valB = Aug_valB[ :, [ 0, 1, 2, 3, 4, 5 ] ]

Aug_valC, _ = SmoothDeriver( Data_val, RegNames, nDerivatives = nDeriv,
                            FilterOrder = FilterOrder, std = std,
                            NormDerivatives = True )
Aug_valC = Aug_valC[ half : -half, : ]
y_valC = Aug_valC[ :, 6 ]
Dc_valC = Aug_valC[ :, [ 0, 1, 2, 3, 4, 5 ] ]

# ------------------------------------------------------------
# 7. Compute errors
# ------------------------------------------------------------
def rel_mae( y, Dc, theta ):
  pred = Dc @ theta
  return ( torch.mean( torch.abs( y - pred ) ) / torch.mean( torch.abs( y ) ) ).item()

print( "=== Approach A: SmoothDeriver as‑is (no scaling) ===" )
print( f"  Coefficients: { [ f'{ c:8.4f}' for c in thetaA.tolist() ] }" )
print( f"  Training rel. MAE: { rel_mae( yA, DcA, thetaA ):.5f}" )
print( f"  Validation rel. MAE: { rel_mae( y_valA, Dc_valA, thetaA ):.5f}" )

print( "\n=== Approach B: SmoothDeriver + time‑scaling (physical units) ===" )
print( f"  Coefficients: { [ f'{ c:8.4f}' for c in thetaB.tolist() ] }" )
print( f"  Training rel. MAE: { rel_mae( yB, DcB, thetaB ):.5f}" )
print( f"  Validation rel. MAE: { rel_mae( y_valB, Dc_valB, thetaB ):.5f}" )

print( "\n=== Approach C: NormDerivatives=True (current tutorial) ===" )
print( f"  Coefficients: { [ f'{ c:8.4f}' for c in thetaC.tolist() ] }" )
print( f"  Training rel. MAE: { rel_mae( yC, DcC, thetaC ):.5f}" )
print( f"  Validation rel. MAE: { rel_mae( y_valC, Dc_valC, thetaC ):.5f}" )

# ------------------------------------------------------------
# 8. Trajectory simulation using physical coefficients (from B)
# ------------------------------------------------------------
cB = thetaB.cpu().numpy()
def reco_ode_B( t, state ):
  x1, v1, x2, v2 = state
  u_t = float( np.interp( t, t_vals, u_vals ) )
  a1 = cB[ 0 ] * x1 + cB[ 1 ] * x2 + cB[ 2 ] * u_t + cB[ 3 ] * v1 + cB[ 4 ] * v2
  a2 = 0.0 # we only fitted mass 1 for simplicity
  return [ v1, a1, v2, a2 ]

sol_rec = spi.solve_ivp( reco_ode_B, [ 0, T ], sol.y[ :, 0 ], t_eval = t_vals, method = 'RK45' )
mse_B = np.mean( ( sol.y[ 0 ] - sol_rec.y[ 0 ] )**2 )
print( f"\nTrajectory MSE x1 (using physical coeffs): { mse_B:.4e}" )

# Also test with unscaled coefficients (A) for comparison
cA = thetaA.cpu().numpy()
def reco_ode_A( t, state ):
  x1, v1, x2, v2 = state
  u_t = float( np.interp( t, t_vals, u_vals ) )
  a1 = cA[ 0 ] * x1 + cA[ 1 ] * x2 + cA[ 2 ] * u_t + cA[ 3 ] * v1 + cA[ 4 ] * v2
  return [ v1, a1, v2, 0.0 ]
sol_recA = spi.solve_ivp( reco_ode_A, [ 0, T ], sol.y[ :, 0 ], t_eval = t_vals, method = 'RK45' )
mse_A = np.mean( ( sol.y[ 0 ] - sol_recA.y[ 0 ] )**2 )
print( f"Trajectory MSE x1 (unscaled coeffs): { mse_A:.4e}" )
