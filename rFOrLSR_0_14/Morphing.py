"""
This place is a message... and part of a system of messages... pay attention to it! Sending this message was important to us. We considered ourselves to be a powerful culture.
This place is not a place of honor...no highly esteemed deed is commemorated here... nothing valued is here.

What is here is dangerous and repulsive to us. This message is a warning about danger.

The danger is in a particular location... it increases toward a center... the center of danger is here... of a particular size and shape, and below us.

The danger is still present, in your time, as it was in ours. The danger is to the body, and it can kill.
The form of the danger is an emanation of energy.
The danger is unleashed only if you substantially disturb this place physically. This place is best shunned and left uninhabited.
"""

import torch as tor
import tqdm
import itertools as it
import scipy.optimize as spop # for the minimization via Trust Region Methods

# Internal imports
from .CTors import Combinations
from . import HelperFuncs as HF


#################################################################################################### Optimizer helper functions ############################################
## ************************************************************************************************ Optimizer: Cost function ********************************************************************************
def DMFunc( yO, Xl, ksi, PA, f ):
  '''Correlation, assumes that yO is already centered and normed'''
  C_ksi = PA @ f( Xl @ ksi ) # Orthogonalize
  C_ksi -= tor.mean( C_ksi ) # Center
  return ( ( ( yO.T @ C_ksi )**2 )[0,0] / tor.sum( C_ksi**2 ) ) # correlation, [0,0] to flatten the ( 1,1 ) array to a float



# ************************************************************************************************ Optimizer: Gradient ********************************************************************************
def DMGrad( yO, Xl, ksi, PA, f, fPrime ):
  # yO -= tor.mean( yO ); yO /= tor.sum( yO**2 ) # security check
  Xl_ksi = Xl @ ksi
  C_ksi = PA @ f( Xl_ksi ); C_ksi -= tor.mean( C_ksi ) # Centered and orthogonalized vector
  Corr = ( yO.T @ C_ksi ) /tor.sum( C_ksi**2 )
  
  return ( 2*Corr * Xl.T @ ( fPrime( Xl_ksi )*( PA.T @ ( yO - Corr * C_ksi ) ) ) )



# ************************************************************************************************ Optimizer: Hessian ********************************************************************************
def DMHessian( yO, Xl, ksi, PA, f, fPrime, fSecond ):
  '''Non-linearity agnostic Hessian for the dictionary morphing cost function.
  Shape assumptions: yO → ( p,1 ), Xl → ( p,r ), ksi → ( r,1 ), PA → ( p,p ). Wrong if yO →(p,) due to implicit broadcasting'''
  
  # Pre-computations
  Xl_ksi = Xl @ ksi # used by all derivatives of the function
  f1 = fPrime( Xl_ksi ) # first derivative vector
  C = PA @ f( Xl_ksi ); C -= tor.mean( C ) # Orthogonalize and center
  C_norm2 = tor.sum( C**2 ) # ||C( ksi )||_2^2
  QCorr = yO.T @ C / C_norm2 # Quasi-Correlation <yO; C( ksi )> Projection of orthonormal gradient on ground thruth
  CQCorr = QCorr * C # used twice
  
  # Addition decomposed into its Terms, themselves decomposed into the their Gram-Matrix components
  A = ( ( ( yO - 2*CQCorr ).T @ PA ) * f1.T ) @ Xl # first line
  A = A.T @ A * 2/C_norm2
  
  B = ( Xl.T*( ( ( yO - CQCorr ).T @ PA ) * fSecond( Xl_ksi ).T ) ) @ Xl *( 2*QCorr )
  
  D = PA @ ( Xl*f1 )
  D = D.T @ D *( -2*QCorr**2 ) # c already used as letter, mult at the end since smallest matrix
  
  E = ( tor.sum( PA, axis = 0 ) * f1.T ) @ Xl
  E = ( E.T @ E ) *( 2/len( yO ) *QCorr**2 ) # mult at the end since smallest matrix, len( yO )→p

  return ( A + B + D + E ) # Assemble all lines constituing the Hessian



# ************************************************************************************************ Expression Parser ********************************************************************************
def Expressionparser( idx, T ):

  if ( idx>T[-1][0] ): raise ValueError( "The asked index exceeds the highest registered index in T" )
  fs = 0 # selected function
  while ( idx > T[fs][0] ): fs +=1 # Scann tuples to compare the upper bound and exit when needed

  # idx is the matrix column index, while T contains that index%nMonomials - all missing terms. Thus the correct index is retrieved by subtracting the cumsum of all terms before
  CumSum = 0 # number of terms in total before the current fs
  for f in range( fs ): CumSum += len( T[f][1] ) # stop before fs, T[f] = non-lin entry tuple, [1] is the element list

  return ( T[fs][1][idx - CumSum], fs ) # return aactual index i0 and the function index fs



# ************************************************************************************************ Is Morphable ********************************************************************************
def IsMorphable( f, x0, order ):
  '''Function checking if the passed function pointer f points to a numerical function that is not morphable.
  
  ### Inputs:
  - `f`: ( pointer to function ) pointing to a numerical scalar function
  - `x0`: ( signal array ) used to determine the extended range where to check for relaxed homogeneity
  - `order`: ( int ) indicating the morphing order, used to return False if>0 as any function is non-zero order morphable

  ### Output:
  - `IsMorphable`: ( boolean ) being True if the function is morphable and False if not
  '''
  
  R = 5 * tor.max( tor.abs( x0 ) ) # range determining the checked interval w.r.t x0 ( much larger to be sure )
  X = tor.linspace( -R, R, 50, endpoint = True ) # linspace as X axis, spanning x0 range
  fx = f( X ) # precompute as potentially used twice
  if ( tor.allclose( fx, X, 1e-12 ) ): return ( False ) # [R 1/3] identity always excluded
  if ( order > 0 ): return ( True ) # [R 2/3] if terms are added, the function morphs irrespectively of Relaxed Homogeneity
  C = f( 10*X )/fx # arbitrarily take a=2 and compute f( ax )/f( x )=c

  return ( not tor.allclose( C, C[0], 1e-12 ) ) # [R 3/3] elementwise comparison for RH, true if all true



# ************************************************************************************************ Genetic Vectorspace Generator ********************************************************************************
def GenVSGen( f, U, i0, Dc, order, Psi, Psi_n, y, m ):
  '''
  Brute Force Genetic vectorspace generator, which tries all combinations of tuples shorter or equal the morphing order for many coeffcients to determine the best one.

  ### Inputs:
  - `f`: ( pointer to function ) pointing to a numerical scalar function
  - `U`: ( list ) containing all indexes allowed for the morphing
  - `i0`: ( int ) containing the Dc column index of the morphed regressor's argument
  - `Dc`: ( ( p,nC )-sized nd-array ) containing the Dictionary of rergessors
  - `order`: ( int ) indicating the number of terms to add during the vectorspace construction
  - `Psi/_n`: ( ( s,p )-sized nd-array ) containing the orthogonalized ( and normed for Psi_n ) already selected terms. both replace PA for efficiency reasons
  - `y`: ( (p,)-sized nd-array ) containing the system output
  - `m`: ( ( 1,m )-sized nd-array ) containing the mean of Dc's columns
  
  ### Outputs:
  - `tOut`: ( list of ints ) containing the indexes of the Dc columns spanning the optimization vectorspace
  - `ksiOut`: ( list of floats ) containing the linear combination coefficients of tOut
  '''

  if ( Psi is not None ): y = y - Psi_n @ ( Psi.T @ y ) # orthogonalize to eliminate the already explained variance if not first term ( equivalent to PA@y )
  
  UNoi0 = U.copy() # create a copy to remove the ell-th element without damaging the original U
  if ( i0 in UNoi0 ): UNoi0.remove( i0 ) # rFOrLSR could have chosen |reg| while reg is no longer in U

  # baseline metrics
  IMax = f( ( Dc[:,i0] + m[:,i0] ).reshape(-1,1) ); IMax -= tor.mean( IMax )
  if ( Psi is not None ): IMax = IMax - Psi_n @ ( Psi.T @ IMax ) # orthogonalize
  IMax = ( IMax.T @ y )**2 / HF.Norm2( IMax )
  tMax = [i0] # best known result is the baseline
  ksiMax = [1] # the baseline is unscaled

  for O in range( order+1 ): # iterate over all lengths, start at 0 which adds no terms but just a coefficient
    Ksi = tor.tensor( list( it.product( *( ( tor.linspace( 0.3, 2, 13 ), ) + O * ( tor.linspace( -2.5, 2.5, 13 ), ) ) ) ) ).T # Generate the coefficient combination matrix
    if ( f == tor.exp ): Ksi /= 10 # avoids overflow
    for t in tqdm.tqdm( it.combinations( UNoi0, O ), total = Combinations( len( UNoi0 ), O ), desc = "Current Term Morphing Order " + str( O ), leave = False ): # iterate over all combiantions of length O
      t = list( ( i0, *t ) ) # prepend the chi0's index to the tuple, transform to list to index the arrays
      F = f( ( Dc[:, t] + m[:,t] ) @ Ksi ) # Create matrix of selected regressors with all
      F -= tor.mean( F, axis = 0, keepdims = True ) # columnwise centering
      if ( Psi is not None ): F = F - Psi_n @ ( Psi.T @ F ) # decomposed PA ( much faster than PA@F )

      QERR = tor.ravel( ( F.T @ y )**2 / HF.Norm2( F ) ) # Quasi-ERR for all lin. Comb.
      l = tor.argmax( QERR ) # find highest QERR
      if ( QERR[l]>IMax ): IMax = QERR[l]; tMax = t.copy(); ksiMax = tor.copy( Ksi[:,l] ) # if better QERR than known


  tOut = []; ksiOut = []
  for index in range( len( tMax ) ): # length is unknown
    if ( ksiMax[index] != 0 ): tOut.append( tMax[index] ); ksiOut.append( ksiMax[index] )

  return ( tOut, ksiOut ) # return filtered ksi



# ************************************************************************************************ infinitesimal Optimizer ********************************************************************************
def InfOPT( y, Xl, ksiS, Ds, Dc, PA, f, fPrime, fSecond, Reps, A_T, Psi, Psi_n, W_T, L ):
  # TODO entire description with all variables
  '''
  Function refining or correcting the linear combination coefficients outputted by GenVSGen.
  y is assumed mean-free while Xl's columns must have their mean added.
  A_T is a copy of A until all needed entries, the original remains thus unaffected
  '''

  # -------------------------------------------------------------------------- Wrappers for scipy TRM ----------------------------------------------------------------------------------------
  # Wrappers for the optimizers as they require a single argument and must be sign flipped since scipy minimizes
  yo = PA @ y # orthogonalize to eliminate the already explained variance
  def TRM_DMFunc( ksi ):    return ( -1 * DMFunc( yo.reshape(-1,1), Xl, ksi.reshape(-1,1), PA, f ) )
  def TRM_DMGrad( ksi ):    return ( -1 * tor.ravel( DMGrad( yo.reshape(-1,1), Xl, ksi.reshape(-1,1), PA, f, fPrime ) ) )
  def TRM_DMHessian( ksi ): return ( -1 * DMHessian( yo.reshape(-1,1), Xl, ksi.reshape(-1,1), PA, f, fPrime, fSecond ) )

  # ---------------------------------------------------------------------------- Optimizer procedure ----------------------------------------------------------------------------------------
  MinMSE = tor.inf; ksiMin = tor.copy( ksiS ) # default values to be overwritten
  W_T.append( None )

  NoiseAmp = 10 # twice the maximal perturbation for the random retriggering

  for r in range( Reps+1 ): # retrigger the optimizer multiple times ( parallelizable
    if ( r == 0 ): ksiT = tor.copy( ksiS ) # GenVSGen output as baseline to beat
    else: # perform one optimizer iteration
      if ( r == 1 ): u = tor.zeros( ksiS.shape )
      else:
        u = NoiseAmp * ( tor.random.random( ksiS.shape ) - 0.5 ) # generate random vector
        while ( tor.allclose( f( ksiS + u ), tor.zeros( ksiS.shape ), 1e-10 ) ): u = NoiseAmp * ( tor.random.random( ksiS.shape ) - 0.5 ) # generate other starting point
      ksiT = spop.minimize( TRM_DMFunc, tor.ravel( ksiS + u ), jac = TRM_DMGrad, hess = TRM_DMHessian, method = "trust-exact", options = {"gtol": 1e-10} )["x"] # extract only the optimized variable
    
    fT = f( Xl @ ksiT ); fT -= tor.mean( fT ) # evaluate and center

    if ( Psi is None ): # no prior terms exist in the regression ( imposed or selected ), thus don't orthogonalize
      thetaT = fT @ y / HF.Norm2( fT ) # yields scalar
      MSE = fT * thetaT # remains a vector, partial MSE computation

    else: # not the first term in the regression
      PAfT = PA @ fT # orthogonalize
      A_T[:-1,-1] = Psi_n.T @ fT; W_T[-1] = PAfT @ y / HF.Norm2( PAfT ) # Add new regressor to A and reg coeffs to W
      # TODO Replace with torch
      # thetaT = spla.solve_triangular( A_T, W_T, overwrite_b= False, unit_diagonal=True ) # obtain the regression coefficients for validation
      if ( Ds is None ): MSE = tor.column_stack( ( Dc[:, L], fT ) ) @ thetaT # version for vectors, partial MSE computation
      else: MSE = tor.column_stack( ( Ds, Dc[:, L], fT ) ) @ thetaT # version for vectors, partial MSE computation
    
    MSE = tor.mean( ( y - MSE )**2 ) # Mean square error to penalize large errors and ignore small ones
    
    # QERR = tor.column_stack( ( Dc[:,L], fT ) )@thetaT; QERR = ( QERR@y )**2/norm2( QERR ) # Correlation metric
    if ( MSE < MinMSE ): MinMSE = MSE; ksiMin = tor.copy( ksiT ) # store best encountered QERR and ksi
  
  return ( ksiMin )



# ************************************************************************************************ Morphing function ********************************************************************************
def Morpher( U, ell, Psi, Psi_n, y, A_T, W_T, L, Ds, Dc, MDict ):
  ''''Morphing function containing all morphing subfunction and the final data handling.
  Means is not updated since morphable terms are not re-morphed'''
  if ( Psi.shape == ( len( y ), 0 ) ): Psi = None # overwrite for clarity
  
  # Import from outside since overwritten: Dc, Psi, Psi_n, A, W, L, ERR, MetaData, DcNames
  # 1. Parser ( doesn't need to recognize ell>nC since morphed terms are not added to U )
  i0, fs = Expressionparser( U[ell], MDict["T"] )
  
  # 2. Morphability test
  # TODO: this should be replaced by a list, allowing the user to select which terms to morph, rather than morphign everything that can
  Morphable = IsMorphable( MDict["fPtrs"][fs], Dc[:,i0] + MDict["DcMeans"][:,i0], MDict["Order"] ) # takes f, chi0 and the order
  if ( not Morphable ): return ( None )

  # TODO see having already chosen terms creates problems with the morphing

  # 3. GenVSGen
  LM, ksiS = GenVSGen( MDict["fPtrs"][fs], U, i0, Dc, MDict["Order"], Psi, Psi_n, y, MDict["DcMeans"] )
  
  # 4. Infinitesimal Optimizer
  Xl = Dc[:, LM] + MDict["DcMeans"][:, LM] # Continue with GenVSGen output
  if ( Psi is None ): PA = tor.eye( len( y ) )
  else: PA = tor.eye( len( y ) ) - Psi_n @ Psi.T
  ksi = InfOPT( y, Xl, tor.ravel( ksiS ), Ds, Dc, PA, MDict["fPtrs"][fs], MDict["f1Ptrs"][fs], MDict["f2Ptrs"][fs], 10, A_T, Psi, Psi_n, W_T.copy(), L )

  # 5. MetaData
  MorphData = ( Dc.shape[1], fs, LM, ksi )
  Regstr = MDict["fStrs"][fs] + "( " # guaranteed to require () since no identity
  Arg = ""
  for i in range( len( LM ) ):
    Arg += f'{ksi[i]:.3f}' + tor.tensor( MDict["DcNames"] )[LM[i]] + " " # add rounded coeff with the term and a space
    if ( i+1 < len( LM ) and ksi[i+1] >= 0 ): Arg += "+" # add a + if not last index and nto negative
  Regstr += Arg[:-1] + " )" # eliminate the last space and close parenthesis

  Reg = MDict["fPtrs"][fs]( Xl @ ksi ); Reg -= tor.mean( Reg ) # recreate regressor to output
  
  return ( MorphData, Reg, Regstr )