import re # regexp for the RegressorParser
import scipy.signal as sps
import numpy as np
import tqdm
import torch as tor

import matplotlib.pyplot as plt
from matplotlib import patches # for the Unit circle of zPlanePlot
from contextlib import contextmanager
from typing import Optional, Tuple, Sequence, Union, Iterator

@contextmanager
def _dark_style() -> Iterator[ None ]:
  with plt.style.context( 'dark_background' ):
    yield

# Library internal imports from sibling folders
from .. import CTors
from .. import HelperFuncs as HF

################################################################################################################################################
####                                                     (Linear) IIR Analysis Tools                                                       #####
################################################################################################################################################

# ############################################################### FOrLSR to IIR #############################################################
def rFOrLSR2IIR( theta: Sequence[ float ], L: Sequence[ int ], RegNames: Sequence[ str ] ) -> Tuple[ np.ndarray, np.ndarray ]:
  '''Converts the FOrLSR Output into a a,b IIR filter coefficient vectors.
  There is no guarantee on the order of the regressors so some matching is required.
  Additionally, y[k-j] terms are sign flipped (IIR convention).

  ### Inputs:
  - `theta`: (1D float iterable) containing the regression coefficients
  - `L`: (1D int iterable) containing the selected regressor indices
  - `RegNames`: (1D string iterable) containing the regressor names

  ### Outputs:
  - `b`: (1D float np.array) containing the b (numerator) coefficients
  - `a`: (1D float np.array) containing the a (denominator) coefficients
  '''

  # ***************************************************** Input validation *****************************************************
  L = list( L )
  theta = list( theta )

  if ( len( L ) == 0 ):          raise ValueError( "L must not be empty" )
  if ( len( L ) != len( theta ) ): raise ValueError( f"Length mismatch: L has { len( L ) } elements, theta has { len( theta ) }" )

  n_reg_names = len( RegNames )
  for idx in L:
    if ( not ( 0 <= idx < n_reg_names ) ): raise ValueError( f"Regressor index { idx } out of bounds (0..{ n_reg_names - 1 })" )

  if ( len( set( L ) ) != len( L ) ): raise ValueError( "Duplicate regressor indices in L are not allowed" )

  # Explicitly reject y[k] (zero-delay denominator term)
  for idx in L:
    if ( RegNames[ idx ] == 'y[k]' ): raise ValueError( "y[k] term is not allowed; denominator a[0] is fixed to 1" )

  # ***************************************************** Regressor parsing *****************************************************
  def RegressorParser( Term: str ) -> Tuple[ str, int ]:
    Pattern = re.compile( r'([xy])\[(k-\d+)\]' ) # matches x[k-j] or y[k-j], j >= 1

    if ( Term == 'x[k]' ): return ( 'x', 0 )

    Match = Pattern.match( Term )
    if ( Match ): # Extract the variable (x or y) and the delay (j)
      Variable, DelayStr = Match.groups()
      Delay = int( DelayStr.split( '-' )[ 1 ] )
      return ( Variable, Delay )

    raise ValueError( f"Invalid term: { Term }. Only linear IIR terms x[k] or x[k-j]/y[k-j] (j>=1) are allowed." )

  # Parse all selected regressors
  CoeffList = []
  for i in range( len( theta ) ):
    Variable, Delay = RegressorParser( RegNames[ L[ i ] ] )
    CoeffList.append( ( theta[ i ], Variable, Delay ) )

  # Check for duplicate (variable, delay) combinations after parsing
  Seen = set()
  for _, var, d in CoeffList:
    pair = ( var, d )
    if ( pair in Seen ): raise ValueError( f"Duplicate regressor mapping found: ({ var }, delay { d })" )
    Seen.add( pair )

  # ***************************************************** Build IIR vectors *****************************************************
  MaxLag = max( delay for _, _, delay in CoeffList )
  a = np.zeros( MaxLag + 1, dtype = float )
  a[ 0 ] = 1.0
  b = np.zeros( MaxLag + 1, dtype = float )

  for val, var, Delay in CoeffList:
    if ( var == 'x' ): b[ Delay ] = val
    elif ( var == 'y' ): a[ Delay ] = -val # sign flip because y terms move to the other side in the difference equation

  return ( b, a )


#   ############################################################### IIR Spectrum #############################################################
def IIR_Spectrum( b_a_List: Optional[ Sequence[ Tuple[ Sequence[ float ], Sequence[ float ] ] ] ] = None,
                 h_List: Optional[ Sequence[ np.ndarray ] ] = None,
                 FilterNames: Optional[ Sequence[ str ] ] = None,
                 Fs: Union[ int, float ] = 44_100,
                 Resolution: int = 5_000,
                 xLims: Optional[ Sequence[ float ] ] = None,
                 yLimMag: Optional[ list ] = None,
                 w_List: Optional[ Sequence[ float ] ] = None
               ) -> Tuple[ plt.Figure, np.ndarray ]:
  '''Plots the magnitude and phase spectrum of the passed IIR-filters.
  The Magnitude response if plotted as: 20 * np.log10( np.maximum( abs( h ), 1e-06 ) ) to avoid zero-division warnings

  ### Inputs:
  - `b_a_List`: List of 2D-tuples containing (b, a) filter coefficients with b and a being iterables
  - `h_List`: List of 1D-iterables containing the complex frequency response
  - `FilterNames`: List of strings containing the filter names
  - `Fs`: (int / float = 44_100) containing the Sampling frequency
  - `Resolution`: (int = 5_000) containing the plot's resolution of the frequency axis
  - `xLims`: (2D-iterable of floats/ints = [1, Fs / 2]) containing the x-axis limits (1 since logarithmic)
  - `yLimMag`: (2D-iterable of floats/ints = [0, 1]) containing the y-axis limits of the magnitude plot
  - `w_List`: (optional) 1D array of frequencies used together with `h_List`; must be same length as each response in `h_List`

  ### Outputs:
  - `Fig`: Figure object containing the plot
  - `Ax`: Axes object containing the plot
  '''
  # ***************************************************** Input checking *****************************************************
  if ( ( b_a_List is None ) and ( h_List is not None ) ): CoeffList = h_List; Coeff_Type = "h"
  elif ( ( b_a_List is not None ) and ( h_List is None ) ): CoeffList = b_a_List; Coeff_Type = "b_a"
  else: raise ValueError( "Either b_a_List or h_List must be passed" ) # covers both None or both not None

  if ( len( CoeffList ) < 1 ): raise ValueError( "CoeffList must contain at least one filter" )
  if ( not isinstance( CoeffList, ( list, tuple ) ) ): raise ValueError( "CoeffList must be a list or tuple" )

  if ( Coeff_Type == "b_a" ):
    if ( not all( isinstance( CoeffList[ i ], ( tuple, list ) ) and len( CoeffList[ i ] ) == 2 for i in range( len( CoeffList ) ) ) ):
      raise ValueError( "b_a_List must be a list of 2D-tuples" )
  else: # Coeff_Type == "h"
    if ( not all( isinstance( CoeffList[ i ], ( np.ndarray, tuple, list ) ) for i in range( len( CoeffList ) ) ) ):
      raise ValueError( "CoeffList must be a list of numpy arrays" )

  if ( type( Fs ) not in [ int, float ] ): raise ValueError( "Fs must be an integer or float" )
  if ( Fs < 1 ): raise ValueError( "Fs must be a positive integer" )
  if ( Resolution < 1 ): raise ValueError( "Resolution must be a positive integer" )
  Resolution = int( Resolution ) # enforce integer

  # Validate custom frequency vector if provided
  if ( Coeff_Type == "h" and w_List is not None ):
    w_List = np.asarray( w_List, dtype = float )
    if ( w_List.ndim != 1 ): raise ValueError( "w_List must be a 1D array" )
    if ( ( np.any( w_List < 0 ) ) or ( np.any( w_List > Fs / 2 ) ) ): raise ValueError( "w_List values must be between 0 and Fs/2" )
    for idx, h in enumerate( CoeffList ):
      if ( len( h ) != len( w_List ) ): raise ValueError( f"h_List[{ idx }] length ({ len( h ) }) does not match w_List length ({ len( w_List ) })" )

  if ( Coeff_Type == "h" and w_List is None ):
    for idx, h in enumerate( CoeffList ):
      if ( len( h ) != Resolution ):
        raise ValueError( f"h_List[{ idx }] length ({ len( h ) }) does not match Resolution ({ Resolution })" )

  if ( FilterNames is None ): FilterNames = [ f'Filter { i + 1 }' for i in range( len( CoeffList ) ) ]
  else:
    if ( len( CoeffList ) != len( FilterNames ) ): raise ValueError( "CoeffList and FilterNames must have the same length" )

  if ( xLims is None ): xLims = [ 1, Fs / 2 ]

  # Handle yLimMag auto vs manual
  if ( yLimMag is None ):
    UpdateLowLim = True
    UpdateHighLim = True
    yLimMag = [ np.inf, -np.inf ] # lowest values to be updated
  else:
    # respect only the user-imposed limits
    if ( yLimMag[ 0 ] is None ):
      UpdateLowLim = True
      yLimMag[ 0 ] = np.inf
    else: UpdateLowLim = False

    if ( yLimMag[ 1 ] is None ):
      UpdateHighLim = True
      yLimMag[ 1 ] = -np.inf
    else: UpdateHighLim = False

  # ************************************************************** Plots **************************************************************
  with _dark_style():
    Fig, Ax = plt.subplots( 2, 1, sharex = True )
    Ax[ 0 ].set_title( 'Frequency Response' ) # randomly Ax[0], Ax[1] also valid

    # Prepare magnitude response
    Ax[ 0 ].set_xlabel( 'Frequency [Hz]' )
    Ax[ 0 ].set_xlim( xLims )
    Ax[ 0 ].set_ylabel( 'Magnitude [dB]' )
    # yLim is set after the loop, as there the min and max can be updated
    Ax[ 0 ].grid( which = 'both', alpha = 0.2 )

    # Prepare phase response
    Ax[ 1 ].set_xlabel( 'Frequency [Hz]' )
    Ax[ 1 ].set_ylabel( 'Phase [Radians]' )
    Ax[ 1 ].grid( which = 'both', alpha = 0.2 )

    # Plot the actual frequency responses
    for coeffs in CoeffList:
      if ( Coeff_Type == "b_a" ): w, h = sps.freqz( coeffs[ 0 ], coeffs[ 1 ], worN = Resolution, fs = Fs )
      else: # Coeff_Type == "h"
        if ( w_List is not None ): w = w_List
        else:                  w = np.linspace( 0, Fs / 2, Resolution, endpoint = True )
        h = np.asarray( coeffs, dtype = complex )

      Magnitude = 20 * np.log10( np.maximum( np.abs( h ), 1e-06 ) )

      if ( UpdateLowLim ):  yLimMag[ 0 ] = np.min( ( yLimMag[ 0 ], np.min( Magnitude ) ) )
      if ( UpdateHighLim ): yLimMag[ 1 ] = np.max( ( yLimMag[ 1 ], np.max( Magnitude ) ) )

      # Mask DC to avoid log(0) warning and missing point on semilogx
      mask = w > 0
      Ax[ 0 ].semilogx( w[ mask ], Magnitude[ mask ] )
      Ax[ 1 ].semilogx( w[ mask ], np.unwrap( np.angle( h ) )[ mask ] )

    # add small margins for aesthetics
    if ( UpdateLowLim ):  yLimMag[ 0 ] -= 2
    if ( UpdateHighLim ): yLimMag[ 1 ] += 2

    # Final sanity check: lower must be less than upper
    if ( yLimMag[ 0 ] >= yLimMag[ 1 ] ): raise ValueError( f"Final magnitude limits are invalid: lower={ yLimMag[ 0 ] }, upper={ yLimMag[ 1 ] }" )

    Ax[ 0 ].set_ylim( yLimMag[ 0 ], yLimMag[ 1 ] )

    Ax[ 0 ].legend( FilterNames )
    Ax[ 1 ].legend( FilterNames )

    Fig.tight_layout()

  return ( Fig, Ax )


################################################################ zPlanePlot #############################################################
def zPlanePlot( b: Sequence[ float ], a: Union[ float, Sequence[ float ] ] = 1, Title: Optional[ str ] = None ) -> Tuple[ np.ndarray, np.ndarray, float ]:
  '''Plot the poles and zeros of the passed filter in the z-plane.

  ### Inputs:
  - `b`: (1D array-like) numerator coefficients
  - `a`: (Optional: 1D array-like or scalar) denominator coefficients (default 1)
  - `Title`: (str) figure title

  ### Outputs:
  - `z`: (1D complex np.array) zeros
  - `p`: (1D complex np.array) poles
  - `k`: (float) filter gain (b[0]/a[0]) consistent with freqz
  '''
  # Ensure b and a are at least 1D arrays
  b = np.atleast_1d( np.asarray( b, dtype = float ) )
  a = np.atleast_1d( np.asarray( a, dtype = float ) )

  if ( a[ 0 ] == 0.0 ): raise ValueError( "Denominator leading coefficient a[0] must not be zero" )

  # Compute poles and zeros (scale invariant)
  z = np.roots( b ) # zeros
  p = np.roots( a ) # poles
  k = b[ 0 ] / a[ 0 ] # gain factor consistent with freqz

  # Create figure
  with _dark_style():
    Fig, Ax = plt.subplots()
    if ( Title is not None ): Fig.suptitle( Title )

    # Unit circle
    Ax.add_patch( patches.Circle( ( 0, 0 ), radius = 1, fill = False, ls = 'dashed' ) )

    # Plot zeros and poles
    Ax.plot( z.real, z.imag, 'go', ms = 10 )
    Ax.plot( p.real, p.imag, 'rx', ms = 10 )

    # Cartesian grid
    Ax.spines[ 'bottom' ].set_position( 'zero' ) # move bottom line to center (real axis)
    Ax.spines[ 'left' ].set_position( 'zero' ) # move left line to center (imaginary axis)
    Ax.spines[ 'top' ].set_visible( False ) # hide top line
    Ax.spines[ 'right' ].set_visible( False ) # hide right line
    Ax.grid( which = 'both', alpha = 0.15 )
    Ax.axis( 'scaled' ) # square plot

    # Set limits and ticks: find the most far away zero/pole for scale
    z_max = np.max( np.abs( z ) ) if z.size > 0 else 0.0
    p_max = np.max( np.abs( p ) ) if p.size > 0 else 0.0
    Lim = 0.1 + max( 1.0, z_max, p_max ) # always show unit circle

    Ax.set_xlim( -Lim, Lim )
    Ax.set_ylim( -Lim, Lim )

    # declutter the axis if many ticks. Safeguard for Matplotlib taking ages and tons of RAM
    if ( Lim >= 100 ):  TickSpacing = 100 # pretty instable filter though :P
    elif ( Lim >= 50 ):   TickSpacing = 10
    elif ( Lim >= 25 ):   TickSpacing = 5
    elif ( Lim >= 10 ):   TickSpacing = 2
    elif ( Lim >= 5 ):    TickSpacing = 1
    elif ( Lim >= 1.5 ):  TickSpacing = 0.5
    else:             TickSpacing = 0.25

    nTicks = int( Lim / TickSpacing )
    FurthestTick = TickSpacing * ( nTicks + 1 )
    ticks = np.arange( -FurthestTick, FurthestTick + TickSpacing, TickSpacing )
    Ax.set_xticks( ticks ); Ax.set_yticks( ticks )

    Fig.tight_layout()

  return ( z, p, k )


########################################################################################################################
####                                          Non-Linear Analysis Tools                                            #####
########################################################################################################################
#  Note: these tools are relatively outdated

# ############################################ Variable Selection Procedure ##################################################
# TODO: use same in-place BLAS as rFOrLSR
def ComputeERR( y: tor.Tensor, Ds: tor.Tensor ) -> np.ndarray:
  '''Imposed only part of the rFOrLSR, as only the ERR of imposed terms is computed and returned.

  ### Inputs:
  - `y`: (1D torch.Tensor) containing the system output vector (must be zero‑mean)
  - `Ds`: (2D torch.Tensor) containing the imposed terms

  ### Outputs:
  - `ERR`: (np.array of float) containing the error reduction ratio (ERR) of the imposed terms in the same order as Ds
  '''

  s2y = ( y @ y ).item() # mean free observation empiric variance
  if ( s2y <= 1e-15 ): raise ValueError( "Output y has zero variance (is constant). ERR computation impossible." )

  ERR = np.full( Ds.shape[ 1 ], 0.0, dtype = np.float64 ) # list of Error reduction ratios of all imposed regressors
  Psi = tor.empty( ( len( y ), 0 ) ); Psi_n = tor.empty( ( len( y ), 0 ) ) # Create empty ( p, 0 )-sized matrices to simplify the code below

  if ( Ds.shape[ 1 ] == 0 ): return ERR # R[1/3]

  # First iteration treated separately since no orthogonalization and no entry in A, and requires some reshapes
  Omega = Ds[ :, 0 ]
  n_Omega = max( tor.sum( Omega**2 ).item(), 1e-12 ) # scalar, ≥ 1e-12 (fudge factor)
  Psi = Omega[ :, None ] # (p,1)
  Psi_n = Psi / n_Omega
  ERR[ 0 ] = ( ( Psi_n.T @ y ).item() )**2 * n_Omega / s2y

  # ------------------------------- remaining columns -----------------------------------------------------
  for col in range( 1, Ds.shape[ 1 ] ):
    if ( np.sum( ERR[ : col ] ) >= 1.0 ): return ( ERR ) # R[2/3] Early exit if max ERR reached, array is init with 1s

    Omega = Ds[ :, col ] - Psi_n @ ( Psi.T @ Ds[ :, col ] ) # orthogonalize only the current column ( no reshape needed )
    n_Omega = max( tor.sum( Omega**2 ).item(), 1e-12 ) # squared euclidean norm of Omega or fudge factor

    ERR[ col ] = ( ( Omega @ y ).item() / n_Omega )**2 * n_Omega / s2y

    # store the orthogonalised regressor
    Psi = tor.column_stack( ( Psi, Omega ) )
    Psi_n = tor.column_stack( ( Psi_n, Omega / n_Omega ) )

  return ( ERR ) # R[3/3]


# ############################################################################ Expansion Order Estimator #############################################################################
def ExpansionOrderEstimator( x: tor.Tensor, y: tor.Tensor, MaxLags: Tuple[ int, int ] = ( 15, 15 ),
                            MaxOrder: int = 5,
                            VarianceAcceptThreshold: float = 0.98,
                            Plot: bool = True
                          ) -> Tuple[ Optional[ int ], np.ndarray ]:
  '''Variable selection function determining the required Monomial expansion order for y and x for rFOrLSR dictionary sparcification.

  This function is NARMAX specific since lagged variables are checked using arbitrary-order polynomial NARX models rather than Taylor expansions.

  Note: For model order > 2, this function might be a lot slower than an Arbo with a large dictionary, so use only for analysis or for Dcs not fitting in memory.

  ### Inputs:
  -`x`: (1D torch.Tensor) containing the system input vector
  -`y`: (1D torch.Tensor) containing the system output vector
  -`MaxLags`: (2D int Tuple) containing the maximum lags for n_b and n_a, defaults to 30 for both ( n_b, n_a )
  -`MaxOrder`: (int > 0) Maximum approximation order used for the estimation
  -`VarianceAcceptThreshold`: ( float ) the minimum explained variance of the NARMAX expansion to estimate the needed delays

  ### Output:
  - `ModelOrder`: (int or None) The chosen expansion's order, or None if the threshold was not met.
  - `ModelExplainedVariance`: ( (ModelOrder+1,)-shaped np.array of float) The chosen model's ERR sum (percentage of explained variance).
                               The 0-th entry corresponds to the constant model (0% explained variance).
  '''

  # Defensive programming
  if ( x.shape != y.shape ): raise AssertionError( "x and y must have the same shape. Note that both are flattened for processing" )
  if ( ( x.ndim != 1 ) or ( y.ndim != 1 ) ): raise AssertionError( "x or y is not a (p,)-shaped Tensor" )
  if ( ( MaxOrder < 1 ) or ( not isinstance( MaxOrder, int ) ) ): raise AssertionError( "MaxOrder must be an int >= 1" )

  y: tor.Tensor = tor.ravel( y )
  x: tor.Tensor = tor.ravel( x )

  # A) Model order evaluation -------------------------------------------------
  ModelExplainedVariance = [ 0.0 ] # Summed ERR of all models orders. Start at 0 to represent the 0th order model, being a constant. The optimal constant is the mean of y being 0
  y_cut, RegMat, RegNames = CTors.Lagger( ( x, y ), MaxLags )
  y_cut -= y_cut.mean()
  ProgressBar = tqdm.tqdm( desc = "Currently analyzed expansion order", unit = "" ) # Initialise progressbar without giving the max to have a counter

  ModelOrder = None
  for ModelOrder in range( 1, MaxOrder + 1 ):
    ProgressBar.update()
    RegMatTMP = CTors.Expander( RegMat, RegNames, ExpansionOrder = ModelOrder )[ 0 ]

    # ComputeERR returns an Array of ERR and stops upon the first ERR > 1 entry and fills the rest of the array with 0, so re-clip the sum since %
    ModelErr = min( 1.0, np.sum( ComputeERR( y_cut, RegMatTMP - RegMatTMP.mean( axis = 0, keepdims = True ) ) ) )
    ModelExplainedVariance.append( ModelErr )

    if ( ModelExplainedVariance[ -1 ] >= VarianceAcceptThreshold ): break # found sufficient order
  ProgressBar.close()

  if ( ModelExplainedVariance[ -1 ] < VarianceAcceptThreshold ): # loop finished without reaching the threshold
    print( "\nThe VarianceAcceptThreshold was not met, increase MaxOrder and/or MaxLags" )
    ModelOrder = None # return None when threshold not reached
  else: print( f"An order { ModelOrder } model explaining { 100 * ModelExplainedVariance[ -1 ]:.2f} % of the variance was selected.\n" )

  ModelExplainedVariance = np.array( ModelExplainedVariance )

  # B) Plotting -----------------------------------------------------------------
  if ( Plot ):
    with _dark_style():
      Fig, Ax = plt.subplots()
      Ax.plot( 100 * ModelExplainedVariance )
      Ax.set_xticks( np.arange( len( ModelExplainedVariance ) ) )
      Ax.set_xticklabels( np.arange( len( ModelExplainedVariance ) ) )
      Ax.grid( which = 'both', alpha = 0.15 )
      Ax.axhline( y = 100 * VarianceAcceptThreshold, c = 'purple', linewidth = 1.5, linestyle = '--' )
      Ax.legend( [ "Model Explained Variance", "User Variance Acceptance Threshold" ] )
      Ax.set( title = f"Model Order Estimation using MaxLags: { MaxLags }",
                   xlabel = "Model Expansion Order", ylabel = "Explained Variance [%]" )
      Fig.tight_layout()

  return ( ModelOrder, ModelExplainedVariance )


# ############################################################################ Variable Selection Procedure #############################################################################
def MaxLagsEstimator( x: tor.Tensor, y: tor.Tensor, ModelOrder: int,
                     MaxLags: Tuple[ int, int ] = ( 15, 15 ),
                     VarianceAcceptThreshold: float = 0.98,
                     Plot: bool = True,
                     SaveFig: Optional[ str ] = None
                   ) -> Tuple[ Optional[ dict ], np.ndarray ]:
  '''Variable selection function determining the maximum lags for y and x for rFOrLSR 
  dictionary sparcification. This function is NARMAX specific since lagged variables are
  checked using arbitrary-order polynomial NARX models rather than Taylor expansions.
  Everything in purple on the plot is below the VarianceAcceptThreshold.

  Note: For model order > 2, this function might be a lot slower than an Arbo with a large dictionary, so use only for analysis or for Dcs not fitting in memory.
  Note: If any of the recommended lags contain the maximum lag as passed by the user, then the passed lags are not sufficient and a warning will be printed.

  ### Inputs:
  -`x`: (1D torch.Tensor) containing the system input vector
  -`y`: (1D torch.Tensor) containing the system output vector
  -`ModelOrder`: (int > 0) Polynomial expansion order used for the estimation
  -`MaxLags`: (2D int Tuple) containing the maximum lags for n_b and n_a (thus x, y), defaults to 15 for both
  -`VarianceAcceptThreshold`: ( float ) the minimum explained variance of the NARMAX expansion to estimate the needed delays
  -`Plot`: (bool) if True, the plot will be generated
  -`SaveFig`: (str) path to save the plot, only works if Plot = True

  ### Output:
  - `Grid`: ( MaxLags[1]+1, MaxLags[0]+1 )-shaped np.array containing the ERR values displayed by the plot.
            Rows correspond to y‑lags (na), columns to x‑lags (nb).
  - `Recommendations`: (Dict or None) containing the optimal lags with the system with the minimal x & y, x, y lags,
                       or None if no configuration reaches the required threshold.
  '''

  # Defensive programming
  if ( x.shape != y.shape ):             raise AssertionError( "x and y must have the same shape. Note that both are flattened for processing" )
  if ( ( x.ndim != 1 ) or ( y.ndim != 1 ) ): raise AssertionError( "x or y is not a (p,)-shaped Tensor" )
  if ( ( ModelOrder < 1 ) or ( not isinstance( ModelOrder, int ) ) ): raise AssertionError( "MaxOrder must be an int >= 1" )
  if ( SaveFig is not None ):
    if ( not Plot ): raise AssertionError( "SaveFig can only be used if Plot = True" )
    SaveFig = SaveFig.replace( "\\", "/" ) # simple normalisation

  y = tor.ravel( y )
  x = tor.ravel( x )

  # A) Model computation ------------------------------------------------------
  print( f"\nComputing the Grid with maximum lags at ({ MaxLags[ 0 ] }, { MaxLags[ 1 ] }) and for a model order of { ModelOrder }:" )
  Grid = np.full( ( MaxLags[ 1 ] + 1, MaxLags[ 0 ] + 1 ), np.nan ) # rows = na, cols = nb
  ProgressBar = tqdm.tqdm( total = Grid.size )
  tol: float = 1e-11 # tolerance for “already explained” optimisation

  for na in range( MaxLags[ 1 ] + 1 ):
    for nb in range( MaxLags[ 0 ] + 1 ):
      # Optimisation: if both previous cells already explained everything, copy the value.
      if ( ( Grid[ max( na - 1, 0 ), nb ] >= 1.0 - tol ) and ( Grid[ na, max( nb - 1, 0 ) ] >= 1.0 - tol ) ):
        Grid[ na, nb ] = 1.0
      else:
        y_cut, RegMat, RegNames = CTors.Lagger( Data = ( x, y ), Lags = ( nb, na ) )
        RegMat, RegNames = CTors.Expander( RegMat, RegNames, ExpansionOrder = ModelOrder )

        ERRArray = ComputeERR( y_cut - y_cut.mean(),
                                      RegMat - RegMat.mean( axis = 0, keepdims = True ) )
        Grid[ na, nb ] = min( 1.0, np.sum( ERRArray ) )
      ProgressBar.update()
  ProgressBar.close()

  # B) Lags recommendation ----------------------------------------------------
  if ( np.max( Grid ) < VarianceAcceptThreshold ):
    print( "\nWARNING: The passed MaxLags don't suffice for the desired variance\n" )
    Recommendations = None # return None when no valid configuration exists
  else:
    Recommendations = {}
    valid = Grid > VarianceAcceptThreshold

    # Min_XY: smallest na+nb
    idx = np.argwhere( valid )
    sums = idx[ :, 0 ] + idx[ :, 1 ]
    best_idx = np.argmin( sums )
    best_na, best_nb = idx[ best_idx ]
    Recommendations[ "Min_XY" ] = ( best_nb, best_na ) # (x-lags, y-lags)

    # Min_Y: smallest nb for the smallest possible na
    for na in range( MaxLags[ 1 ] + 1 ):
      col_valid = np.where( Grid[ na, : ] > VarianceAcceptThreshold )[ 0 ]
      if ( col_valid.size > 0 ):
        best_nb = col_valid[ 0 ]
        Recommendations[ "Min_Y" ] = ( best_nb, na )
        break

    # Min_X: smallest na for the smallest possible nb
    for nb in range( MaxLags[ 0 ] + 1 ):
      row_valid = np.where( Grid[ :, nb ] > VarianceAcceptThreshold )[ 0 ]
      if ( row_valid.size > 0 ):
        best_na = row_valid[ 0 ]
        Recommendations[ "Min_X" ] = ( nb, best_na )
        break

  # C) Plot -------------------------------------------------------------------
  if ( Plot ):
    with _dark_style():
      ColorBarMax = np.max( Grid )
      if ( ColorBarMax <= VarianceAcceptThreshold ): ColorBarMax = 1.0 # make everything purple (clipped)

      Fig, Ax = plt.subplots()
      Im = Ax.pcolormesh( Grid, cmap = 'viridis', edgecolors = 'k', linewidth = 2, vmin = VarianceAcceptThreshold, vmax = ColorBarMax )

      DotSize = 60
      if ( Recommendations is not None ):
        Ax.scatter( Recommendations[ "Min_Y" ][ 0 ] + 0.5, Recommendations[ "Min_Y" ][ 1 ] + 0.5, color = 'r', s = DotSize )
        Ax.scatter( Recommendations[ "Min_X" ][ 0 ] + 0.5, Recommendations[ "Min_X" ][ 1 ] + 0.5, color = 'r', s = DotSize )
        Ax.scatter( Recommendations[ "Min_XY" ][ 0 ] + 0.5, Recommendations[ "Min_XY" ][ 1 ] + 0.5, color = 'r', s = DotSize )

      Ax.set( title = f"Model Order: { ModelOrder }",
                   xlabel = "x[k-i] regressors", ylabel = "y[k-i] regressors",
                   xlim = ( 0, MaxLags[ 0 ] + 1 ), ylim = ( 0, MaxLags[ 1 ] + 1 ),
                   xticks = ( 0.5 + np.arange( MaxLags[ 0 ] + 1 ) ),
                   yticks = ( 0.5 + np.arange( MaxLags[ 1 ] + 1 ) ),
                   xticklabels = ( np.arange( MaxLags[ 0 ] + 1 ) ),
                   yticklabels = ( np.arange( MaxLags[ 1 ] + 1 ) ),
                   aspect = 'equal' )
      Fig.colorbar( Im )
      Fig.tight_layout()

      if ( SaveFig is not None ): plt.savefig( SaveFig )

  return ( Recommendations, Grid )
