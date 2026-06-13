import torch as tor
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from NARMAX.CTors import SmoothDeriver

p = 3000
Fs = 100.0
dt = 1.0 / Fs
T = p * dt

m1, m2 = 1.0, 1.0
k1, k2 = 100.0, 50.0
c1, c2 = 2.0, 1.0

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

a1_true = ( -c1 * v1 - k1 * x1 - k2 * ( x1 - x2 ) - c2 * ( v1 - v2 ) + u_sig ) / m1
a2_true = ( -c2 * ( v2 - v1 ) - k2 * ( x2 - x1 ) ) / m2

Data = tor.column_stack( [ x1, x2, u_sig ] )
RegNamesIn = [ "x1", "x2", "u" ]

FilterOrders = [ 5, 7, 11, 15, 21, 31, 41, 51, 61, 81, 101 ]
Stds = [ 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50 ]

results = []
total = len( FilterOrders ) * len( Stds )
idx = 0

for fo in FilterOrders:
  for s in Stds:
    idx += 1
    if ( idx % 20 == 0 or idx == total ): print( f"Processing { idx }/{ total }..." )
    AugData, _ = SmoothDeriver( Data, RegNamesIn, nDerivatives = 2,
                                    FilterOrder = fo, std = s, NormDerivatives = True )
    d_x1 = AugData[ :, 3 ]
    d_x2 = AugData[ :, 4 ]
    d2_x1 = AugData[ :, 6 ]
    d2_x2 = AugData[ :, 7 ]

    skip = fo // 2
    lo, hi = skip, len( d_x1 ) - skip

    def pearson_r( a, b ):
      a_c = a - a.mean()
      b_c = b - b.mean()
      r = ( a_c * b_c ).sum() / ( tor.sqrt( ( a_c**2 ).sum() ) * tor.sqrt( ( b_c**2 ).sum() ) + 1e-12 )
      return r.item()

    r_v1 = pearson_r( d_x1[ lo : hi ], v1[ lo : hi ] )
    r_v2 = pearson_r( d_x2[ lo : hi ], v2[ lo : hi ] )
    r_a1 = pearson_r( d2_x1[ lo : hi ], a1_true[ lo : hi ] )
    r_a2 = pearson_r( d2_x2[ lo : hi ], a2_true[ lo : hi ] )

    mean_r = np.mean( [ r_v1, r_v2, r_a1, r_a2 ] )
    min_r = min( r_v1, r_v2, r_a1, r_a2 )
    results.append( ( fo, s, mean_r, min_r, r_v1, r_v2, r_a1, r_a2 ) )

results.sort( key = lambda r: -r[ 2 ] )

header = f"{ 'FilterOrder':>11s}  { 'std':>6s}  { 'mean_r':>7s}  { 'min_r':>7s}  { 'r_v1':>7s}  { 'r_v2':>7s}  { 'r_a1':>7s}  { 'r_a2':>7s}"
sep = "-" * len( header )

print( "\n" + "=" * 80 )
print( " TOP 15 BEST PARAMETER COMBINATIONS (by mean correlation)" )
print( "=" * 80 )
print( header )
print( sep )
for i in range( 15 ):
  fo, s, mn, mi, r1, r2, r3, r4 = results[ i ]
  print( f"{ fo:>11d}  { s:>6.3f}  { mn:>7.5f}  { mi:>7.5f}  { r1:>7.5f}  { r2:>7.5f}  { r3:>7.5f}  { r4:>7.5f}" )

print( "\n" + "=" * 80 )
print( " BOTTOM 3 WORST COMBINATIONS" )
print( "=" * 80 )
print( header )
print( sep )
for i in range( -1, -4, -1 ):
  fo, s, mn, mi, r1, r2, r3, r4 = results[ i ]
  print( f"{ fo:>11d}  { s:>6.3f}  { mn:>7.5f}  { mi:>7.5f}  { r1:>7.5f}  { r2:>7.5f}  { r3:>7.5f}  { r4:>7.5f}" )

best = results[ 0 ]
print( "\n" + "=" * 80 )
print( " OPTIMAL PARAMETERS" )
print( "=" * 80 )
print( f"  FilterOrder = { best[ 0 ] }" )
print( f"  std         = { best[ 1 ]:.3f}" )
print( f"  mean r      = { best[ 2 ]:.6f}" )
print( f"  min r       = { best[ 3 ]:.6f}" )
print( f"  r(v1)       = { best[ 4 ]:.6f}" )
print( f"  r(v2)       = { best[ 5 ]:.6f}" )
print( f"  r(a1'')     = { best[ 6 ]:.6f}" )
print( f"  r(a2'')     = { best[ 7 ]:.6f}" )
print()

# --- Heatmap ---
fo_vals = sorted( set( r[ 0 ] for r in results ) )
std_vals = sorted( set( r[ 1 ] for r in results ) )
Z = np.full( ( len( std_vals ), len( fo_vals ) ), np.nan )
for fo, s, mn, *_ in results:
  i = fo_vals.index( fo )
  j = std_vals.index( s )
  Z[ j, i ] = mn

with plt.style.context( 'dark_background' ):
  fig, ax = plt.subplots( figsize = ( 12, 6 ) )
  pc = ax.pcolormesh( fo_vals, std_vals, Z, cmap = 'plasma', shading = 'auto' )
  cb = fig.colorbar( pc, ax = ax, label = 'Mean correlation r' )
  ax.set_xlabel( 'FilterOrder' )
  ax.set_ylabel( 'std' )
  ax.set_title( 'Grid Search: Mean Correlation (shape matching)' )
  fig.tight_layout()
  fig.savefig( 'grid_search_correlation.png', dpi = 150 )
  plt.close( fig )

import os
print( f"Saved: grid_search_correlation.png ({ os.path.getsize( 'grid_search_correlation.png' ) } bytes)" )
