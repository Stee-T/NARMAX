"""This submodule contains function and signal constructors for some common fitting types :
- `Lagger`: Creates delayed version of the signal. (x → x[k-j], j in some set )
- `Expander`: Creates monomial expansions of terms with products or powers of passed terms (x1, x2 → x1^2*x2)
- `Non-Linearizer`: apply a list of non-linearities (x→f(x)) and make them rational if desired (x → 1/f(x))
"""

import numpy as np
import torch as tor
import itertools as it

# ####################################################################################### Helper Functions #############################################################################################
def Combinations( N, k ):
  '''From N chose k for positive values only'''
  if ( k > N ): return ( 0 ) # property of the operator
  if ( k < 0 or N < 0 ): raise ValueError( "N or k is negative, which is not supported" )

  def factorial( n ):
    fact = 1 # base case, guaranteed output even if looop not entered
    for i in range( 2,int( n )+1 ): fact *= i # multiply upwards, +1 due to python indexing
    return ( fact )

  return ( int( factorial( N ) / ( factorial( k ) * factorial( N - k ) ) ) ) # N!/( k!*( N-k )! )


# ####################################################################################### CTors #############################################################################################


# ************************************************************************************ Regression matrix CTor ********************************************************************************
def Lagger( Data, MaxLags ):
  '''Function returning a matrix containing all polynomial terms up to the given na, nb and if noise given nc cut to contain only swung in states.
  
  Note: x should be centered before sending it through the system. y and e should not be centered before being passed to this function.

  ### Input:
  - `Data ( x, y [,e] )`: ( 2/3D Tuple of (p,)-shaped Tensor ) ) containing the input
  - `MaxLags ( nb, na [,nc] )`: ( 2/3D Tuple of ints ) ) containing the maximum lags of each terms
  - `Func`: ( None/ list of str ) allowing to have the function in the variable name before performing the poly expansion per signal in Data
  
  ### Output:
  - `y`: (torch.tensor) containing the cut and centered system output
  - `Regressors`: ( torch.tensor ) where each column is a regressor over the swung-in dataset lengh ( p-q )
  - `RegressorNames`: ( np.array ) containing the regressor names
  '''
  
  # Bullshit prevention
  if ( len( Data ) != len( MaxLags ) ):         raise AssertionError( "The numbers of signals and maximum lags don't correspond" )
  
  for i in range( 1, len( Data ) ): 
    if ( len( Data[0] ) != len( Data[i] ) ):    raise AssertionError( "All Signals must have the same lenght" )
    if ( not tor.isfinite( Data[i] ).all() ):   raise AssertionError( "Passed data contains inf or nans. The to be fitted system is unstable in general or for this particular sequence. Try a new one" )

  for i in range( len( MaxLags ) ):
    if ( not isinstance( MaxLags[i], int ) ):   raise AssertionError( "All MaxLags elements must be integers" )
  
  # ------------------------------------------------------------------------------------------- A) Initialization --------------------------------------------------------------------------------------
  q = max( MaxLags ) # detect maximum lag
  p = Data[0].view( -1, 1 ).shape[0] # number of samples in the signals
  nSignals = len( MaxLags ) # Used to determine if ( x, y ) or ( x, y, e ) tuple is passed
  nSimpleTerms = np.sum( MaxLags ) + nSignals - 1 # + 1 due to x[k] and if included +1 for e[k]
  RegMat = tor.full( ( p-q, nSimpleTerms ), tor.nan ) # Regressor matrix big enough for all simple and optionally combined swung-in terms
  RegNames = [] # Regressor Names
  
  # ------------------------------------------------------------------------------------------- B) Delayed-copy constructors ---------------------------------------------------------------------------
  # Simple terms ( exclude y[k], not being part of the poly expansion to preserve system causality ), thus if qy = 0, no y terms are created
  for col in range( 1, MaxLags[1] + 1 ): # Add the y[k-i] terms, starting at 1 to exclude y[k] term
    RegMat[:, col-1] = Data[1].view( -1 )[ q - col : p - col ] # flatten and take needed slice
    RegNames.append( f"y[k-{ col }]" )

  for col in range( MaxLags[0] + 1 ): # Add the x[k-i] terms, +1 for x[k]
    RegMat[:, MaxLags[1] + col] = Data[0].view( -1 )[ q - col : p - col ]
    RegNames.append( f"x[k-{ col }]" )
  RegNames[ MaxLags[1] ] = "x[k]" # overwrite x[k-0], sits at MaxLags[1]=n_a, since zero-based
 
  if ( nSignals == 3 ): # Same procedure for the noise terms
    FirstEk = MaxLags[0] + MaxLags[1] + 1 # first noise term being e[k], sits at MaxLags[0]+[1]+1=na+nb+1, since zero-based
    for col in range( MaxLags[2] + 1 ):
      RegMat[:, FirstEk + col] = Data[2].view( -1 )[ q - col : p - col ]
      RegNames.append( f"e[k-{ col }]" ) # Add the e[k-i] terms, +1 for e[k]
    RegNames[FirstEk] = "e[k]" # overwrite e[k-0]


  return ( Data[1].view( -1 )[q:p], RegMat, np.array( RegNames ) ) # Data[1][q:p] is the cut y vector


# **************************************************************************************** Monomial Expansion *********************************************************************************************
def Expander( Data, Names, ExpansionOrder, IteractionOnly = False ):
  ''' Creates a Monomial expansion from the passed Data and column names on the GPU.

  Note: This function is just a combinatorial CTor. Expressions of the type f( phi[k-i]*phi[k-j] ) are generated by applying f( • ) on the output ( NonLinearizer CTor ),
  while expressions of the type f( phi1 )*f( phi2 ) are obtained by applying f( • ) on the regressors before passing them to this function.

  ### Inputs:
  - `Data`: ((p, nC) pytorch Tensor) containing columnwise the regressors to be expanded
  - `Names`: ((nC,) Iterable) containing the column names
  - `ExpansionOrder`: (int) Monomial expansion Order , dictating maximally how many terms are multiplied with each other in each combination
  - `IteractionOnly`: (bool) if True, only composite terms are generated like (x[k] * x[k-1]) and no powers of a single term like x[k]^2

  ### Outputs:
  - `RegMat`: ((p, nRegs) pytorch Tensor) containing the expanded regressors
  - `Names`: ((nRegs,) Iterable) containing the expanded column names
  
  '''
  # Inspired from https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/preprocessing/_polynomial.py#L30
  
  if ( ( ExpansionOrder < 1 ) or ( not isinstance( ExpansionOrder, int ) ) ): raise ValueError( "ExpansionOrder should be an int >= 1" )
  if ( ExpansionOrder == 1 ): return ( Data, Names ) # Nothing happens for 1-order polynomial expansion

  if ( Data.ndim != 2 ): raise ValueError( "Data should be 2-dimensional with regressors as columns" )
  if ( Data.shape[1] != len( Names ) ): raise ValueError( "Number of names does not match number of columns in the Data" )

  # ------------------------------------------------------------------------------- A) Regressor Computation ---------------------------------------------------------------
  # Compute the total number of Regressors and pre-allocate memory
  if ( IteractionOnly ): nRegs = np.sum( [ Combinations(Data.shape[1], o) for o in range( 1, ExpansionOrder + 1 ) ] )
  else: nRegs = Combinations( Data.shape[1] + ExpansionOrder, ExpansionOrder ) - 1
  RegMat = tor.full( ( Data.shape[0], nRegs ), np.nan )

  # degree 1 Expansion terms (no changes)
  RegMat[:, :Data.shape[1]] = Data
  index = list( range( Data.shape[1] ) )
  currentCol = Data.shape[1]
  index.append( currentCol )

  for _ in range( 2, ExpansionOrder + 1 ): # loop over degree >= 2 terms, is skipped if ExpansionOrder = 1
    new_index = []
    end = index[-1]

    for feature_idx in range( Data.shape[1] ):
      start = index[feature_idx]
      new_index.append( currentCol )
      if ( IteractionOnly ): start += index[feature_idx + 1] - index[feature_idx]
      next_col = currentCol + ( end - start ) # next column index being current + current lenght
      if ( next_col <= currentCol ): break # don't overshoot
      
      # RegMat[:, start:end] are terms of degree d - 1 that exclude feature feature_idx.
      RegMat[:, currentCol:next_col] = RegMat[:, start:end] * Data[:, feature_idx : feature_idx + 1] # elementwise multiplication
      currentCol = next_col

    new_index.append( currentCol )
    index = new_index

  # ------------------------------------------------------------------------------- B) Names construction ---------------------------------------------------------------
  RegNames = [None] * nRegs # Pre-allocate memory for speed

  Comb = it.combinations if IteractionOnly else it.combinations_with_replacement # chose correct combinations function
  iter = it.chain.from_iterable( Comb( range( Data.shape[1] ), i ) for i in range( 1, ExpansionOrder + 1 ) ) # no need for [] since itertools take variable number of arguments

  # tuples are sorted, being made of sorted containers, thus one can simply count the number of duplicate elements for the powers
  for counter, idx_tuple in enumerate( iter ): # iterate over all combinations of regressors
    Str = "" # reinitialize the string
    IndexSet, PowerSet = np.unique( idx_tuple, return_counts = True ) # remove duplicates and count them

    for idx in range( len( IndexSet ) ):
      Str += f"{ Names[ IndexSet[idx] ] } " if ( PowerSet[idx] == 1 ) else f"{ Names[ IndexSet[idx] ] }^{ PowerSet[idx] } "
    
    RegNames[counter] = Str[:-1] # Cut last space
  
  return ( RegMat, RegNames )


# ************************************************************************************************ Regressor Matrix Transform ********************************************************************************
def NonLinearizer( y, Data, RegNames, Functions, MakeRational = None ):
  ''' Applies the list of passed functions elementwise on the passed data and relabels the columns to take that into account if the first two arguments are not [].
  If no functions are passed and if MakeRational is None, nothing will happen adn a warning is printed
  
  ### Input:
  - `y`: (None, torch Tensor) containing the centered system output. 'None' is only legal if MakeRational is None
  - `Data`: (torch Tensor) containing the regressors in columns
  - `RegNames`: ((nc,)-dimensional ndarray of str) containing the regressor names
  - `Functions`: (list of NonLinearity objects) containing the functions to apply to the data
  - `MakeRational`: (list of bools) containing a True for each function to be make rational
  
  ### Output:
  - `Data`: ( 1, ( len(Functions) + sum(Rational) )*len( Data ) - some entries )- shaped Pandas dict Transformed and renamed data with eliminated duplicates
  - `RegNames`: ((nc,)-dimensional ndarray of str) containing the regressor names
  - `M`: (list of ints ) containing the indices of the applied functions for each created term. Length: Data.shape[1] * ( len(Functions) + sum(MakeRational) )
  '''
  # ---------------------------------------------------------------------------------------------- Bullshit prevention -------------------------------------------------------------------------------------
  # Data-type tests
  if ( not isinstance( Data, tor.Tensor ) ):        raise AssertionError( "The Input data must be a torch.Tensor" )
  if ( not isinstance( Functions, list ) ):         raise AssertionError( "The 'Functions'argument name must be a list of function pointers" )
  
  if (Functions[0].get_Name() != "id"):             raise AssertionError( "The first function in the Functions list must be 'id' per convention" )

  # Length tests
  if ( MakeRational is not None ): # check first since None has no length
    if ( len( Functions ) != len( MakeRational ) ): raise AssertionError( "The length of MakeRational doesn't match that of Functions" )

  if ( ( len( Functions ) == 1 ) and MakeRational is None): # only contains id
    print( "WARNING: No transformations (Functions) or MakeRational instructions were passed, which is sus as this CTor will not do anything" )

  y = y.view( -1 ) # flatten as security
  if ( len( y ) != Data.shape[0] ):                 raise AssertionError( "y's length does not match the Regressors' length" )
  
  if ( MakeRational == [] ): MakeRational = None
  # ---------------------------------------------------------------------------------------- A) Pre-Processing & B) Transformations ----------------------------------------------------------------------------

  # A) Set to list to easily append then concatenate data
  nRegs = Data.shape[1]
  DataList = [ Data ]
  NameList = [ RegNames ]
  M = [0] * nRegs # Morphing meta data containing the index of the applied non-linearity (id for all un-processed terms)

  # B) Compute the transformations and append to the list to finally horizontally concatenate into a single matrix  
  for func in range( 1, len( Functions ) ): # start at 1 to ignore the identity function, skips the loop if Func only contains identity
    DataList.append( Functions[func].get_f()( DataList[0] ) ) # apply the function on the entire passed RegMat
    NameList.append( [ Functions[func].get_Name() + "(" + col + ")" for col in NameList[0] ] ) # Create the list of new names and apply directly
    M += [ func ] * nRegs # tag all regressors to have been processed with that function
  
  # ---------------------------------------------------------------------------------------------- B) Rational Functions -------------------------------------------------------------------------------------

  if ( MakeRational is not None ):
    for func in range( len( Functions ) ):
      if ( MakeRational[func] ): # if the function is to be made rational (contains bool)
        DataList.append( - y.view( -1, 1 ) * DataList[func] ) # 1/ done via multiplication with -y
        M += [ 0 ] * nRegs # tag all rational terms as unmorphable at the moment
        
        RatNames = [None] * nRegs # Init empty list of right length
        # this is ambiguous with terms which are really 1/(...) but yeah whatever
        if ( Functions[func].get_Name() == "id" ): # equivalent to func == 0
          for col in range( len( RatNames ) ): RatNames[col] = "1/(" + NameList[0][col] + ")" # 1/(Reg) for identity
        else:
          for col in range( len( RatNames ) ): RatNames[col] = "1/(" + Functions[func].get_Name() + "(" + NameList[0][col] + "))" # 1/(func(Reg)) for functions
        
        NameList.append( RatNames ) # apply new column names

  return ( tor.hstack( DataList ), np.concatenate( NameList ), M ) # Concatenate the list of segments into a single matrix


####################################################################################### Ultra Orthogonal fitting Stuff ###########################################################################

# Legacy code, as hasn't been updated to new API, etc

# ************************************************************************************** Differentiated Mollifier CTor ********************************************************************************
# def DiffedMollifiersCTor( FilterOrder = 31, nDerivatives = 2, std = 0.12, Plot = False ):
#   '''
#   Constructor function generating the mollifier ( gaussian PDF but other could be implemented ) and all its derivatives up to the given order.
#   The outputted lists contain [FilterOrder]+1 elements where the index corresponds to the filter order, since the 0th element is the function itself ( 0th order derivative ).
  
#   ### Input:
#   -`FilterOrder`: ( int ) giving the number of same used for the FIR's IR = FIR order. Should be odd to have the peak on a sample ( else 2 neighboring samples will be equally weighted )
#   -`nDerivatives`: ( int ) giving the number and thus maximum order of the desired derivatives
#   -`std`: ( float ) standard deviation of the gaussian function
#   -`Plot`: ( bool ) determining if the filters are to be plotted
  
#   ### Output:
#   -`Coeffs`: ( list of ( FilterOrder, )-ndarrays ) containing the filter coeffcients = Impulse responses. Index = derivative order.
#   -`Functions`: ( list of UnivariateSpline functions ) containing representing the splines. Index = derivative order.
#   '''
#   if ( not isinstance( std, float ) ): raise TypeError( "std must be a float" ) # other two variables throw error in linspace or Univariate spline if not ints
  
#   Functions = []; Coeffs = [] # Functions are Spline classes, coefficients are the FIRs
#   X = tor.linspace( 0, 1, FilterOrder )
#   Mollifier = 1/( std * tor.sqrt( 2*tor.pi ) ) * tor.exp( -( X-0.5 )**2 / ( 2*std**2 ) ) # Gauss function have the mean in the IR's middle

#   Coeffs.append( Mollifier/tor.sum( Mollifier**2 ) ) # Append normed mollifier as first FIR, no abs since 100% positive.
#   Functions.append( spi.UnivariateSpline( X, Coeffs[-1], k=3, s=0 ) ) # Fit the Spline on the data, s=0→use all points
  
#   for order in range( 1, nDerivatives+1 ): # Order 0 is the Spline representing Mollifier, thus start at 1 and +1 to not exclude that index ( range )
#     Functions.append( Functions[0].derivative( order ) ) # Store the derivative as function ( spline object )5
#     Coeffs.append( Functions[-1]( X )/tor.sum( tor.abs( Functions[-1]( X ) )**2 ) ) # Compute samples and norm to sum=1
#     Coeffs[-1][[0, -1]] = 0 # Force first and last sample to 0. Automatically the case for large FilterOrder, but not necessarily for small ones
    
#   if ( Plot ):
#     Fig, Ax = plt.subplots()
#     for i in range( nDerivatives+1 ): Ax.plot( Coeffs[i] )
#     Ax.legend( ["Spline"] + ["$\partial_x^" + str( i ) +"$ Spline" for i in range( 1, nDerivatives+1 )] )
  
#   return ( Coeffs, Functions )

# ************************************************************************************************ Derivative augment ********************************************************************************
# def DerivativeAugment( Vector, Matrix, RegCoeffs = [1,1], nDerivatives=2, FilterOrder=31 ):
#   # TODO: this needs to get the pndas dependency removed
#   '''Transformation function concatenating the [nDerivatives] first derivatives of the passed Vector and/or Matrix ( columnswise ) for Data augmentation
#   or Ultra-orthognal Least squares fitting in Sobolev spaces. The function is not in-place to preserve the origianl data.
  
#   ### Input:
#   -`Vector`: ( pd.Series or (p,)-Tensor or None ) containing data to be transformed
#   -`Matrix`: ( pd.Dataframe or ( p,n )-Tensor or None ) containing data to be transformed columnwise
#   -`RegCoeffs`: ( iterable of len=nDerivatives ) containing the regularization coefficients weighting the derivative by scaling that vector segment.
#   this allow to control the contribution of each derivative to the cost function. Applied to the mollifier for efficiency.
#   -`nDerivatives`: ( int ) determining the highest order of derivatives to be computed
#   -`FilterOrder`: ( int ) determining the number of coefficients used by the mollifier, larger values reduce the Hf content more
  
#   ### Output:
#   -`Vector`: ( pd.Series or (p,)-Tensor ) containing the data with appended derivatives
#   -`Matrix`: ( pd.Dataframe or ( p,n )-Tensor ) containing the data with columnwise appended derivatives
#   '''
  
#   if ( len( RegCoeffs ) != nDerivatives ): raise AssertionError( "There is not the same amount of Regularisation coefficients and derivatives to apply them to" )
  
#   FIRs = DiffedMollifiersCTor( FilterOrder, nDerivatives, std=0.12, Plot=False )[0] # take only the impulse response and eliminate the 0th derivative
#   Window = [tor.exp( tor.linspace( -4,0, int( FilterOrder/2 ) ) )]; Window.append( tor.flip( Window[0] ) ) # create exponential fade-in and fade out ( =flipped fade-in )
  
#   if ( Vector is not None ):
#     if ( isinstance( Vector, pd.Series ) ): # exteract information from Series
#       IsSeries = True # needed since Vector gets overwritten by its content as tor.tensor
#       Name = Vector.name # extract the column name
#       Vector = [Vector.to_numpy()[0]] # extract values and put in list ( to_numpy somehow puts the content array inside an array, thus [0] )
#     else: IsSeries = False # passed vector is a numpy array
    
#     for derivative in range( 1, nDerivatives+1 ):      
#       SmoothedDerivative = tor.convolve( Vector[0], RegCoeffs[derivative-1]*FIRs[derivative], "same" ) # apply differentiated Mollifier as overlap-add convolution for efficiency
      
#       # Do fades to mitigate the convolution introduced artefacts at start & end, equivalent to a window but much more efficient avoiding tons of *1 between fades
#       SmoothedDerivative[:int( FilterOrder/2 )] = SmoothedDerivative[:int( FilterOrder/2 )] * Window[0] # fade-in
#       SmoothedDerivative[-int( FilterOrder/2 ):] = SmoothedDerivative[-int( FilterOrder/2 ):] * Window[1] # fade-out
#       Vector.append( SmoothedDerivative ) # append to concatenation list
    
#     Vector = tor.concatenate( Vector ) # vertically concatenate the derivatives to the respective entries
#     if ( IsSeries ): Vector = pd.Series( Vector, Name ) # If was a series to start with re-create a series
  
#   if ( Matrix is not None ): 
#     if ( isinstance( Matrix, pd.DataFrame ) ): # exteract information from Dataframe
#       IsFrame = True
#       Cols = Matrix.columns # list of strign containing column names
#       Matrix = [Matrix.values] # extract the matrix values as array
#     else: IsFrame = False
      
#     # TODO: vectorize this operation by applying the convolution columnwise:
#     #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html
#     #   https://stackoverflow.com/questions/59878951/convolving-a-each-row-of-a-2d-matrix-with-a-vector
#     # TODO enfoice centering on the newly created matrices to avoid bias
    
#     for derivative in range( 1, nDerivatives+1 ):
#       Matrix.append( tor.zeros( Matrix[0].shape ) ) # insert an empty matrix
#       for col in range( Matrix[0].shape[1] ): 
#         SmoothedDerivative = tor.convolve( Matrix[0][:,col], RegCoeffs[derivative-1]*FIRs[derivative], "same" ) # apply differentiated Mollifier on the first Matrix in the list being the utorrocessed one.
#         # Do fades to mitigate the convolution introduced artefacts at start & end, equivalent to a window but much more efficient avoiding tons of *1 between fades
#         SmoothedDerivative[:int( FilterOrder/2 )] = SmoothedDerivative[:int( FilterOrder/2 )] * Window[0] # fade-in
#         SmoothedDerivative[-int( FilterOrder/2 ):] = SmoothedDerivative[-int( FilterOrder/2 ):] * Window[1] # fade-out
#         Matrix[-1][:, col] = SmoothedDerivative
#     Matrix = tor.concatenate( Matrix, axis=0 ) # merge list to single matrix
#     if ( IsFrame ): Matrix = pd.DataFrame( Matrix, columns=Cols ) # retransform into dataframe 
  
#   return ( Vector, Matrix ) # vertically concatenate the derivatives to the respective entries