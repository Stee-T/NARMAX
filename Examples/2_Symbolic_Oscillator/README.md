<div align="justify">

# Example 2: The Symbolic Oscillator

<br/>

This example / tutorial illustrates the use of the Symbolic Oscillator:
- How to construct and format the data required by the object ([Section 3](#3-symbolic-oscillator-inputs))
- How to create a `NARMAX.SymbolicOscillator` object to represent an arbitrary symbolic NARMAX system ([Section 4](#4-simulation-and-processing))
- How to process data with the SymbolicOscillator object ([Section 4](#4-simulation-and-processing))
- How to modulate the regression parameters $\underline{\theta}$ ([Section 4](#4-simulation-and-processing))
- How the function initializing it properly is implemented in the library ([Section 6.1](#61-proper-initialization))
- A short list on all member functions of the `NARMAX.SymbolicOscillator` object ([Section 6.2](#62-list-of-member-functions))

<br/>

## 1. Introduction

Let's start with a few explanations about what the Symbolic Oscillator (SymbOsc) even is. It's a python-object, internally compiled by the library to allow the use of user-defined symbolic NARMAX systems for data generation and processing. In Machine Learning / Statistic-terms, it's basically the "executable" object containing the fitted model.

To illustrate, after the arborescence is run to find a suitable system ([see Tutorial 1](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/1_Linear_in_the_Parameters)), the user has a list of regressors (such as $x_1\[k\]$, $\sin(x_1\[k-1\])$, etc.) and a list of regression coefficients ($\theta_0$, $\theta_1$, $\theta_2$, etc.) which represent the NARMAX system. This is the full solution to the fitting problem, but it requires the user to code the NARMAX system manually to use it either in production or in further testing pipelines. This can be cumbersome, and prevents the automatic deployment and testing of the NARMAX system by requiring human intervention.  
Furthermore, the arborescence itself uses the `SymbolicOscillator` during the (potentially user-provided) validation step to simulate the system to compute the error metrics via actual measurements, rather than approximations like most SotA algorithms currently do. Further library updates / papers will also add the ability to generate the NARMAX systems for better fitting inside the arborescence itself.

The `NARMAX.SymbolicOscillator` object, thus, allows the user to create a symbolic NARMAX system from the provided lists of regressors (list of strings), regression coefficients (float array) and regression non-linearities (lists of function pointers). The SymbOsc object has an internal parsing engine that analyzes the provided lists, checks for correctness, and forwards it to the internal compiler, which will generate the necessary code for the symbolic NARMAX system with optimal CPU/GPU dispatching (currently only partially implemented, being part of my next paper).

This tutorial demonstrates how to use a `NARMAX.SymbolicOscillator`-object to represent and evaluate arbitrary [1] symbolic NARMAX systems, aka applying the system to incommnig data or generating data itself. This example consists of a MISO-system (Multiple-Input Single-Output) with two inputs ($x_1$ and $x_2$), a single output ($y$) and multiple nonlinearities in addition to a supplementary input channel ($W$) for noise, DC-offsets, etc. The to-be-compiled system is

$y\[k\] = W\[k\] + \frac{{\theta_0\frac{y\[k-1\]}{x_2\[k\]} + \theta_1 \text{UFunc}(x_1\[k-1\]) + \frac{\theta_2}{\text{abs}\left( 0.2x_1\[k-1\] + 0.5\cos\left( x_1\[k-2\]x_2\[k\] \right) - 0.2 \right)} }}{1 + \theta_3 x_1\[k-1\]x_2\[k-1\] + \theta_4 x_2^2\[k-2\] + \theta_5\cos\left( 0.2x_1\[k-3\]x_2\[k-1\] + \sin\left( x_2\[k-2\] + 0.5 \right) - 0.1 \right) }$

where $\text{UFunc}(x) := \max(0, \sin(2x))$ is a custom user-defined function, which demonstrates that the provided compiler handles any function passed by the user correctly.

[1] The parser now supports arbitrary nesting depth of functions, as demonstrated by the nested `cos(sin(...))` expression in this example.

<br/>

## 2. Data Generation

This sections explains the creation of the data used in this example.

**Input Signals:** Two input signals, $x_1$​ and $x_2$​, are generated as random tensors with a zero-mean uniform distribution, an amplitude of `InputAmplitude` and a length of `p`.

```python
p: int = 2_000 # dataset size
InputAmplitude: float = 1.5 # Set Input amplitude

x1: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 ) # Mean not subtracted, since no rFOrLSR
x2: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )
```

<br/>

**Additional Input**: This random tensor is constructed as above and just represents an additional additive input to the system to simulate noise, DC-offsets, etc.

```python
AdditionalInput: tor.Tensor = 0.2 * ( tor.rand( p ) - 0.5 )
```

<br/>

**Regression Coefficients:** theta represents the scaling coefficients of the system's regression terms. This would be for example $a$, $b$ in an equation of the form $y = ax^2 + bx$.  
To demonstrate the parameter modulation functionality of the `SymbolicOScillator`, three different sets of coefficients are defined, to be applied on  three (`ThirdBuffer`-long) segments of the dataset.

```python
theta: list[ tor.Tensor ] = [ tor.tensor( [ 0.2,  -0.3,  1,   0.8, -0.3, 1 ] ),
                              tor.tensor( [ 0.25, -0.25, 0.8, 0.9, -0.5, 0.95 ] ),
                              tor.tensor( [ 0.3,  -0.4,  0.7, 0.7, -0.4, 0.9 ] )
                            ]
ThirdBuffer: int = p // 3 # How long we keep the same theta values
```

<br/>

**Nonlinear User Function:** This example's system includes a half-wave rectified sine wave $\text{UserFunction}(x)=\max⁡(0,\sin⁡(2x))$ to represent an arbitrary user-defined nonlinearity for demonstration purposes.

**System:** The `System` function applies the provided system equation (made of hardcoded regressors and a variable regression coefficient array `theta`) to the input signals to generate the output signal. Further, to simplify the for-loop applying the system (see next section), start and end indices are provided to do the demonstrated modulation. The `W` argument accepts an additional buffer to handle `AdditionalInput`. As aforementionned, this allows to add any unprocessed additive input to the system, which can be used to model input noise / supplementary regressors, DC-offsets or whatever your heart desires.

```python
def UserFunction( x: tor.Tensor ) -> tor.Tensor:
  return ( tor.max( tor.tensor( [ 0 ] ), tor.sin( 2 * x ) ) ) # whatever really

def System( y: tor.Tensor, x1: tor.Tensor, x2: tor.Tensor, W: tor.Tensor, theta: tor.Tensor,
            StartIdx: int, EndIdx: int ) -> tor.Tensor:
  for k in range( StartIdx, EndIdx ):
    y[k] = W[k] + ( ( theta[0] * y[k-1] / x2[k] + theta[1] * UserFunction( x1[k-1] )
                      + theta[2] / tor.abs( 0.2 * x1[k-1] + 0.5 * tor.cos( x1[k-2] * x2[k] ) - 0.2 ) ) # Numerator
                                                    /
                    ( 1 + theta[3] * x1[k-1] * x2[k-1] + theta[4] * x2[k-2]**2
                      + theta[5] * tor.cos( 0.2 * x1[k-3] * x2[k-1] + tor.sin( x2[k-2] + 0.5 ) - 0.1 ) ) ) # Denominator
  return ( y )
```

<br/>

## 3. Symbolic Oscillator Constructor Parameters
This section lists and illustrates all arguments the `NARMAX.SymbolicOscillator` constructor accepts, acting as the docs for the CTor.

### 3.1 Non-Optional Parameters:

#### 3.1.1 InputVarNames
List of strings containing the names of the input variables. This is used by the parsing engine to recognize what is a valid variable in the passed expressions.  
Example:

```python
InputVarNames: list[ str ] = [ 'x1', 'x2', 'y' ]
```

#### 3.1.2 NonLinearities
List of `NARMAX.NonLinearity` objects, each representing a nonlinearity used in the system. The constructor of `NARMAX.NonLinearity` class performs the necessary checks and assignments to confirm that the non-linearity is valid. This is used by the compiler to embed the user-defined non-linearity in the generated code.

```python
NonLinearities: list[ NARMAX.NonLinearity ] = [ NARMAX.Identity, # Obligatory for AOrLSR, here optional
                                                NARMAX.NonLinearity( "uFunc", f = UserFunction ),
                                                NARMAX.NonLinearity( "abs", f = tor.abs ),
                                                NARMAX.NonLinearity( "cos", f = tor.cos ),
                                                NARMAX.NonLinearity( "sin", f = tor.sin ),
                                              ] # Used Functions
```

#### 3.1.3 Expressions
List of strings containing the symbolic expressions for the system regressors. These expressions are parsed, verified for syntactic validy then compiled into the SymbolicOscillator object.

**Note 1 (Rational Expressions):** Since the library also supports fitting rational NARMAXes (a NARMAX divided by another NARMAX), one must differentiate between fractional and denominator regressors, such as for example 1/(1+x) as a regressor and (1+x) in the denominator. Thus, the tilde (~) is used to differentiate between both: 1/(1+x) is a fractional regressor, while ~/(1+x) is "1+x" in the denominator of a fractional NARMAX. For an illustration see the example above, which contains both types of regressors.

**Note 2 (Function Names):** Functions used in expression-strings must use the EXACT same name (string) as used in the `NARMAX.NonLinearities` list, otherwise the parser will complain that it doesn't know what you're talking about.

```python
Expressions: list[ str ] = ["y[k-1]/x2[k]", "uFunc( x1[k-1] )", "1/abs( 0.2*x1[k-1] + 0.5*cos( x1[k-2]*x2[k] ) - 0.2 )", # Numerator
                            "~/(x1[k-1]*x2[k-1])", "~/(x2[k-2]^2)", "~/cos( 0.2*x1[k-3]*x2[k-1] + sin( x2[k-2] + 0.5 ) - 0.1 )" ] # Denominator
```

#### 3.1.4 Theta
torch.Tensor of initial regression coefficients. These regression coefficients are used as the default values when the system is compiled. Those can however be changed during runtime, to emulate modulation, as demonstrated in the below use-case example.

### 3.2 The Optional Parameters:

#### 3.2.1 OutputVarName
String that defaults to 'y'. This is useful in general, since it allows the user to name the output variable in something potentially more explicit. It is, however, necessary for MIMO (Multiple-Input Multiple-Output) systems, where there are many output variables, such as for example $y_1$ $y_2$ $y_3$, which the parser doesn't recognize per default as a security measure (this prevents it from misinterpreting the NARMAX's input/output structure).


#### 3.2.2 dtype
"dtype" that defaults to `torch.float64` for maximal precision. Data type for the internal processing and output buffers.


#### 3.2.3 device
String that defaults to the content of the `NARMAX.device` variable, which is set by a procedure trying out `"mps"`, `"cuda"` and if not available defaults to `"cpu"`. The entire library tries to use the GPU/hardware-accelerator whenever possible without requiring any user intervention. However, this variable can be overwritten by the user to change the device it will run on.

<br/>

## 4. Simulation and Processing

**Real System Simulation:** The actual system output ($y$) is simulated using the `System` function across three segments of the dataset (ThirdBuffer), each using different coefficients for theta[i]. Note that to keep this example simple, in the for-loop, the first buffer is truncated (system is zero-initialized).

```python
y: tor.Tensor = tor.zeros( p )
for i in range( 3 ):
  StartIdx: int = ( i * ThirdBuffer ) if ( i > 0 ) else 3 # avoids index error in System
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  y = System( y, x1, x2, AdditionalInput, theta[i], StartIdx, EndIdx ) # only overwrite y[StartIdx:EndIdx]
```
<br/>

**Symbolic Oscillator Prediction:** The `NARMAX.SymbolicOscillator` object is compiled by it's constructor, thus at the object declaration, such that no further step is required by the user.  
The SymbOsc's prediction (`yhat`) is computed using the `Oscillate` member function. As for the hard-coded system, the `Oscillate` function is called across three segments of the dataset (with a length of ThirdBuffer), each using different regressions / model coefficients contained in `theta[i]`. Thus, the input-data to be processed must be segmented into thirds and passed accordingly. For more real-time-like pipelines, the updated variables can of course directly be passed without indexing.
```python
yHat: tor.Tensor = tor.zeros( p )
Model: NARMAX.SymbolicOscillator = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[0] )

for i in range( 3 ):
  StartIdx: int = i * ThirdBuffer # no conditional, since SymbOsc-class handles buffer starts correctly
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  yHat[ StartIdx : EndIdx ] = Model.Oscillate( Data = [ x1[ StartIdx : EndIdx ], x2[ StartIdx : EndIdx ] ],
                                               theta = theta[i], # change regression coefficients
                                               DsData = AdditionalInput[ StartIdx : EndIdx ] # additional input
                                             )
  # Model.set_theta( theta[i] ) # would also work if separating the system update from the processing is desired
```

<br/>

## 5. Error Analysis

```python
with plt.style.context('dark_background'):
    Fig, Ax = plt.subplots()
    Diff: tor.Tensor = (y - yHat)[20:] # cut the start since the system init of the for loops is incomplete
    Ax.plot( Diff.cpu().numpy() ) 
    Ax.set( title = "y - yHat", xlabel = 'k', ylabel = 'y - yHat',
            xlim = ( 0, p-20 ), ylim = ( 1.1*Diff.min().item(), 1.1*Diff.max().item() )
          )
    Ax.grid( which = 'both', alpha = 0.3 )
    Fig.tight_layout()

plt.show()
```

Trivially, this section just plots the difference between the hard-coded system output (`y`) and the symbolic oscillator's prediction (`yhat`) to show that indeed both results are the same (up to floating-point rounding errors).
The first few initial samples are cut away, since the hard-coded system isn't properly initialized to keep the tutorial simple.

<br/>

## 6. Supplementary Material
### 6.1 Proper initialization

As mentioned in the last section, the initialization of a NARMAX system (and thus of an `NARMAX.SymbolicOscillator` object) isn't trivial.
The library therefore provides a convenience-function `NARMAX.InitAndComputeBuffer` to abstract that away from the user, but it is still of interest to understand the procedure.

The `SymbolicOscillator` internally keeps a short rolling window of the most recent past inputs and outputs. These stored values are used whenever a regressor refers to a time index that lies **before** the beginning of the current data buffer (for example `y[k-2]` when `k=0` or `k=1`).

#### 6.1.1 Why Initialization Matters
To exactly reproduce a known output sequence `\underline{y}`, the model's internal state must be filled with the **correct historical data**. The most common mistake is to feed the **first** few sequence samples into the storage. That would only be correct if the required lag window happens to start at index 0, which is not the general case.

Proper initialization guarantees that given a set of input sequences (`Data`), the `NARMAX.SymbolicOscillator` generates the ***EXACT*** same output sequence `y` as the original system (if both equations are the same). This requires particular care, since, per default, the SymbOsc is a zero-initialized "blank system", meaning that all initial conditions are zero.  

To illustrate, be the system $y[k] = a \cdot x[k] + b \cdot x[k-1] + c \cdot y[k-2]$. At time point $k=0$, the two last regressors ($y[k-2]$ and $x[k-1]$) are outside of the current buffer / sequence (negative indices) and are thus assumed to be zero if no other information is available. Those zeros (called initial conditions: $x[-1]$, $y[-1]$ and $y[-2]$) are taken from the object's internal buffers. However, if the "real" system that generated the output sequence $\underline y$ was already "running" before the sequence start, assuming that $y[-1] = 0$, $y[-2] = 0$ and $x[-1]=0$ could result in a completely different output sequence $\hat{\underline y}$. 

This problem is especially pronounced with:
1. **Strongly Auto-Regressive (AR) NARMAX systems**: Errors from the initial conditions are fed back into the system for a potentially long time.
2. **Chaotic or barely stable NARMAX systems**: These can be "knocked-off" from their equilibrium state or start in a completely different part of their phase-plot.

Thus, it is very important to initialize the `NARMAX.SymbolicOscillator` correctly, especially when comparing a `NARMAX.SymbolicOscillator` output to a measurement (e.g., in cost functions or model validation procedures).

#### 6.1.2 Step By Step
One can emulate a correct initialization from an arbitrary buffer by essentially using the buffer-start to fake the initial conditions, as shown in the `InitAndComputeBuffer` function, which we'll work up to in the following example.

First, determine the maximum of the system's input and output lags ($q_x$ and $q_y$), which is how far in teh past the model needs to see to compute the current output. This can be obtained via `get_MaxInputLag` and `get_MaxOutputLag`:

```python
qx: int = Model.get_MaxInputLag()
qy: int = Model.get_MaxOutputLag()
StartIdx: int = max( qx, qy ) 
```

Then, the $q_y$ samples immediately preceding `StartIdx` are taken from the measured system output $\underline y$ and used to overwrite the internal output buffer. The same is done with the set of input sequences $\underline x$. Now the SymbOsc is no longer a "blank" (zero-initialized) system but has valid initial conditions:

```python
# set previous y[k-j] states
Model.set_OutputStorage( y[ StartIdx - qy : StartIdx ].clone() )

# set previous x[k-j] states
Model.set_InputStorage( tor.vstack( [ input[ StartIdx - qx : StartIdx ] for input in Data ] ) )
```

Finally, copy the first `StartIdx` measured output samples $\underline y$ into the output buffer, since those cannot be calculated (they act as the initial conditions). From there on, the SymbOsc can normally run to get the rest of the data:
```python
yHat: tor.Tensor = tor.zeros_like( y )

# take solution samples where Model hasn't got all data. Avoids init-Error spikes
yHat[ : StartIdx ] = y[ : StartIdx ].clone()

# calculate the rest
yHat[ StartIdx: ] = Model.Oscillate( [ input[ StartIdx: ] for input in Data ] )
return ( yHat )
```

#### 6.1.3 The full function

The library helper `InitAndComputeBuffer` consolidates the above logic into a single function, adding basic input validation and support for additive inputs:

```python
from typing import Optional, Sequence
import torch as tor

def InitAndComputeBuffer( Model: object, y: tor.Tensor, Data: Sequence[ tor.Tensor ], DsData: Optional[ tor.Tensor ] = None ) -> tor.Tensor:
  if ( Model.get_MaxPositiveInputLag() > 0 ): raise RuntimeError( "Positive input lags are not yet supported." )

  if ( len( Data ) != Model.get_nInputVars() ): raise ValueError( "Wrong number of input variables." )

  qx: int = Model.get_MaxInputLag()
  qy: int = Model.get_MaxOutputLag()
  StartIdx: int = max( qx, qy )

  # Initial state = immediate past
  Model.set_OutputStorage( y[ StartIdx - qy : StartIdx ].clone() )
  Model.set_InputStorage( tor.vstack( [ input[ StartIdx - qx : StartIdx ] for input in Data ] ) )

  yHat: tor.Tensor = tor.zeros_like( y )
  yHat[ : StartIdx ] = y[ : StartIdx ].clone()

  DataSlice: list[ tor.Tensor ] = [ input[ StartIdx : ] for input in Data ]
  DsSlice: Optional[ tor.Tensor ] = DsData[ StartIdx : ] if ( DsData is not None ) else None

  yHat[ StartIdx : ] = Model.Oscillate( DataSlice, DsData = DsSlice ).to( y.device )
  return yHat
```

**Handling additive inputs (`DsData`)**: Some systems include an extra additive channel (noise, DC offsets, etc.) that is not stored in the internal state (it has no lags). If the true system used such a channel, it must be passed to `InitAndComputeBuffer` via the optional `DsData` argument. The helper will slice it exactly like the input data and forward it to `Oscillate`. Omitting `DsData` when it was present during the original generation will lead to an incorrect output.

**Note:** For analysis purposes, the `NARMAX.SymbolicOscillator` object's internal buffers can be obtained at any time via the member functions `get_OutputStorage` and `get_InputStorage` and overwritten via `set_OutputStorage` and `set_InputStorage`. See the next section for a complete list of member functions.

### 6.2 List of Member Functions

The following list shows all public member functions of the `NARMAX.SymbolicOscillator` object.


#### 6.2.1 set/get_theta
Getter and setter for the regression coefficients

#### 6.2.2 set/get_OutputStorage
Getter and setter for the internal output buffer storing the required past output values (see 6.1)

#### 6.2.3 set/get_InputStorage
Getter and setter for the internal input buffer storing the required past input values (see 6.1)

#### 6.2.4 zeroInternalStorage
Zeros the internal buffers, such that the system isn't influenced by previous buffer data ("blank" system)

#### 6.2.5 get_nRegressors
Returns the number of regressors in the NARMAX expression the object represents

#### 6.2.6 get_MaxInputLag
Returns the largest negative input lag of the NARMAX expression the object represents: the largest j of all x[k-j]

#### 6.2.7 get_MaxPositiveInputLag
Returns the largest positive input lag of the NARMAX expression the object represents. Positive lags allow to represent non-causal systems. It's weird and there are obviously problems at the buffer end (data is needed that doesn't currently exist), but people do what they want, really. Not fully supported though currently.

#### 6.2.8 get_MaxNegOutputLag
Returns the largest negative output lag of the NARMAX expression the object represents. There is no positive output lag, since I really can't predict the future outputs.

#### 6.2.9 oscillate
Processes a list of input sequences, and potentially noise and returns a list of output sequences, as described above.

[Previous Tutorial: 1. Linear in the Parameters](https://github.com/Stee-T/NARMAX/tree/main/Examples/1_Linear_in_the_Parameters)  
[Next Tutorial: 3. Rational Fitting](https://github.com/Stee-T/NARMAX/tree/main/Examples/3_Rational_Fitting)