<div align="justify">

# Example 2: The Symbolic Oscillator

<br/>

This example / tutorial illustrates the Symbolic Oscillator:
- How to construct and format the data required by the object
- How to create a `SymbolicOscillator` object to represent an arbitrary symbolic NARMAX system
- How to process data with the SymbolicOscillator object
- How to modulate the regression parameters $\underline{\theta}$
- How the function initializing it properly is implemented in the library
- A short list on all member functions of the `SymbolicOscillator` object

<br/>

## 1. Introduction

Let's start with a few explanations about what the Symbolic Oscillator (SymbOsc) even is. It's a python-object, internally compiled by the library to allow the use of the user-defined NARMAX systems for data generation and processing. The compilation step also includes an optimal dispatching between GPU and CPU to achieve the maximal performance (I'm writing a paper about it currently). In Machine Learning / statistic-terms, it's basically the "executable" object containing the fitted model.

To illustrate, after the arborescence is run to find a suitable system ([see Tutorial 1](https://github.com/Stee-T/rFOrLSR/tree/main/Examples/1_Linear_in_the_Parameters)), the user has a list of regressors (such as $x_1\[k\]$, $\sin(x_1\[k-1\])$, etc.) and a list of regression coefficients ($\theta_0$, $\theta_1$, $\theta_2$, etc.) which represent the NARMAX system. This is the full solution to the fitting problem, but it requires the user to code the NARMAX system manually to use it either in production or in further testing pipelines. This can be cumbersome, and prevents the automatic deployment and testing of the system by requiring human intervention. Furthermore, the arborescence itself uses the `SymbolicOscillator` during the (potentially user-provided) validation step to simulate the system to compute the error metrics via actual measurements, rather than approximations like most SotA algorithms currently do. Further library updates / papers will also add the ability to generate the NARMAX systems for better fitting inside the arborescence itself.

The `SymbolicOscillator` object, thus, allows the user to create a symbolic NARMAX system from the provided lists of regressors (list of strings), regression coefficients (float array) and regression non-linearities (lists of function pointers). The SymbOsc object has an internal parsing engine that analyzes the provided lists, checks for correctness, and forwards it to the internal compiler, which will generate the necessary code for the symbolic NARMAX system with optimal GPU/CPU dispatching (currently only partially implemented, being part of my next paper).

This tutorial demonstrates how to use a `NARMAX.SymbolicOscillator`-object to represent and evaluate arbitrary [1] symbolic NARMAX systems. This example consists of a MISO-system (Multiple-Input Single-Output) with two inputs ($x_1$ and $x_2$), a single output ($y$) and multiple nonlinearities. The to-be-compiled system is

$y\[k\] = W\[k\] + \frac{{\theta_0\frac{y\[k-1\]}{x_2\[k\]} + \theta_1 \text{UFunc}(x_1\[k-1\]) + \frac{\theta_2}{\text{abs}\left( 0.2x_1\[k-1\] + 0.5x_1\[k-2\]x_2\[k\] - 0.2 \right)} }}{1 + \theta_3 x_1\[k-1\]x_2\[k-1\] + \theta_4 x_2^2\[k-2\] + \theta_5\cos\left( 0.2x_1\[k-3\]x_2\[k-1\] - 0.1 \right) }$

where $\text{UFunc}(x) := \max(0, \sin(2x))$ is a custom nonlinear user-defined function, which demonstrates that the provided compiler handles any function passed by the user correctly.


## 2. Data Generation

```python
p: int = 2_000 # dataset size
InputAmplitude: float = 1.5 # Set Input amplitude

x1: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 ) # Mean not subtracted, since no rFOrLSR
x2: tor.Tensor = ( 2 * InputAmplitude ) * ( tor.rand( p ) - 0.5 )

def UserFunction( x: tor.Tensor ) -> tor.Tensor:
  return ( tor.max( tor.Tensor( [ 0 ] ), tor.sin( 2 * x ) ) ) # Some Weird oscillating ReLU because why not

def System( y, x1, x2, W, theta, StartIdx, EndIdx ) -> tor.Tensor:
  for k in range( StartIdx, EndIdx ):
    y[k] = W[k] + ( ( theta[0] * y[k-1] / x2[k] + theta[1] * UserFunction( x1[k-1] )
                      + theta[2] / tor.abs( 0.2 * x1[k-1] + 0.5 * x1[k-2] * x2[k] - 0.2 ) ) # Numerator
                                                    /
                    ( 1 + theta[3] * x1[k-1] * x2[k-1] + theta[4] * x2[k-2]**2
                      + theta[5] * tor.cos( 0.2 * x1[k-3] * x2[k-1] - 0.1 ) ) ) # Denominator
  return ( y )
```

**Input Signals:** Two input signals, $x_1$​ and $x_2$​, are generated as random tensors with a zero-mean uniform distribution with an amplitude of `InputAmplitude`.

**System:** The `System` function applies the provided system equation (made of hardcoded regressors and a variable regression coefficient array theta) to the input signals to generate the output signal. Further, to simplify the for-loop applying the system (see next section), start and end indices are provided to access the correct parts of the buffer, when theta changes. The W variable allows to add any unprocessed additive input to the system, which can be used to model input noise / supplementary regressors, DC-offset or whatever your heart desires.

**Nonlinear User Function:** The system includes a custom oscillatory ReLU-like nonlinearity $\text{UserFunction}(x)=\max⁡(0,\sin⁡(2x))$ for demonstration purposes.


## 2. Symbolic Oscillator Data
This code segment declares the data required to create the `NARMAX.SymbolicOscillator` object.

```python
InputVarNames: list[ str ] = [ 'x1', 'x2', 'y' ] # Used Variables
NonLinearities: list[ NARMAX.NonLinearity ] = [ NARMAX.Identity, # Obligatory for AOrLSR, here optional
                                                NARMAX.NonLinearity( "uFunc", f = UserFunction ),
                                                NARMAX.NonLinearity( "abs", f = tor.abs ),
                                                NARMAX.NonLinearity( "cos", f = tor.cos ),
                                              ] # Used Functions

Expressions: list[ str ] = ["y[k-1]/x2[k]", "uFunc( x1[k-1] )", "1/abs( 0.2*x1[k-1] + 0.5*x1[k-2]*x2[k] - 0.2 )", # Numerator
                            "~/(x1[k-1]*x2[k-1])", "~/(x2[k-2]^2)", "~/cos( 0.2*x1[k-3]*x2[k-1] - 0.1 )" ] # Used Regressors (1 in deno is implicit)

AdditionalInput: tor.Tensor = 0.2 * ( tor.rand( p ) - 0.5 )

# System / regression coefficients, change to emulate modulation
theta: list[ tor.Tensor ] = [ tor.tensor( [ 0.2,  -0.3,  1,   0.8, -0.3, 1 ] ),
                              tor.tensor( [ 0.25, -0.25, 0.8, 0.9, -0.5, 0.95 ] ),
                              tor.tensor( [ 0.3,  -0.4,  0.7, 0.7, -0.4, 0.9 ] )
                            ]

ThirdBuffer: int = int( p / 3 ) # How long we keep the same theta values
```
The `NARMAX.SymbolicOscillator` class contructor has 4 non-optional parameters:

**InputVarNames:** (non-optional argument) List of strings containing the names of the input variables. This is used by the parsing engine to recognize what is a valid variable in the passed expressions.

**NonLinearities:** (non-optional argument) List of `NARMAX.NonLinearity` objects, each representing a nonlinearity used in the system. The constructor of `NARMAX.NonLinearity` class performs the necessary checks and assignments to confirm that the non-linearity is valid. This is used by the compiler to embed the user-defined non-linearity in the generated code.

**Expressions:** (non-optional argument) List of strings containing the symbolic expressions for the system regressors. These expressions are parsed, verified for syntactic validy then compiled into the SymbolicOscillator object.  
**Note:** Since the library also supports fitting rational NARMAXes (a NARMAX divided by another NARMAX), one must differentiate between fractional and denominator regressors, such as for example 1/(1+x) as a regressor and (1+x) in the denominator. Thus, the tilde (~) is used to differentiate between both: 1/(1+x) is a fractional regressor, while ~/(1+x) is "1+x" in the denominator of a fractional NARMAX. For an illustration see the example above, which contains both types of regressors.

**Theta:** (non-optional argument) torch.Tensor of initial regression coefficients. These regression coefficients are used as the default values when the system is compiled. Those can however be changed during runtime, to emulate modulation, as demonstrated in the below use-case example.

**OutputVarName:** (optional argument) String that defaults to 'y'. This is useful in general, since it allows the user to name the output variable in something potentially more explicit. It is, however, necessary for MIMO (Multiple-Input Multiple-Output) systems, where there are many output variables, such as for example $y_1$ $y_2$ $y_3$, which the parser doesn't recognize per default as a security measure (this prevents it from misinterpreting the NARMAX's input/output structure).

**dtype:** (optional argument) "dtype" that defaults to torch.float64 for maximal precision.

**device:** (optional argument) String that defaults to the content of the `NARMAX.device` variable, which is set by a procedure trying out "mps", "cuda" and if not available defaults to "cpu". The entire library tries to use the GPU/hardware-accelerator whenever possible.

## 3. Simulation and Processing

```python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ y = Real System
y: tor.Tensor = tor.zeros( p )
for i in range( 3 ):
  StartIdx: int = ( i * ThirdBuffer ) if ( i > 0 ) else 3 # avoids index error in System
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  y: tor.Tensor = System( y, x1, x2, AdditionalInput, theta[i], StartIdx, EndIdx ) # only overwrite y[StartIdx:EndIdx]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ yHat = Symbolic Oscillator
yHat: tor.Tensor = tor.zeros( p )
Model: NARMAX.SymbolicOscillator = NARMAX.SymbolicOscillator( InputVarNames, NonLinearities, Expressions, theta[0] )

for i in range( 3 ):
  StartIdx: int = i * ThirdBuffer # no conditional, since NARMAX-class handles buffer starts correctly
  EndIdx: int = ( i + 1 ) * ThirdBuffer
  yHat[ StartIdx : EndIdx ] = Model.Oscillate( Data = [ x1[ StartIdx : EndIdx], x2[ StartIdx : EndIdx ] ],
                                               theta = theta[i], # change regression coefficients
                                               DsData = AdditionalInput[ StartIdx : EndIdx ] # additional input
                                             )
  # Model.set_theta( theta[i] ) # would also work if separating the system update from the processing is desired
```

**Real System Simulation:** The actual system output ($y$) is simulated using the `System` function across three segments of the dataset (ThirdBuffer), each using different coefficients for theta[i]. Note that to keep this example simple, in the for-loop, the first buffer is truncated (system is zero-initialized).

**Symbolic Oscillator Prediction:** The `NARMAX.SymbolicOscillator` object is compiled by it's constructor, thus at the object declaration, such that no further step is required by the user.  
The SymbOsc's prediction (`yhat`) is computed using the `Oscillate` member function. As for the hard-coded system, the `Oscillate` function is called across three segments of the dataset (with a length of ThirdBuffer), each using different regressions / model coefficients contained in `theta[i]`. Thus, the input-data to be processed must be segmented into thirds and passed accordingly. For more real-time-like pipelines, the updated variables can of course directly be passed without indexing.



## 4. Error Analysis

```python
Fig, Ax = plt.subplots()
Diff: tor.Tensor = (y - yHat)[20:] # cut the start since the system init of the foor loops is incomplete
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

## 5. Supplementary Material
### 5.1 proper initialization

As mentioned in the last section, proper initialization of a NARMAX system (and thus of an `NARMAX.SymbolicOscillator` object) must be done correctly. The library therefore provides a convenience-function to abstract that away from the user but it's of interest to understand the procedure. Here is the current implementation (called via `NARMAX.InitAndComputeBuffer`):

```python
def InitAndComputeBuffer( Model: NARMAX.SymbolicOscillator, y: tor.Tensor, Data: list[tor.Tensor] ):
  '''Helper function initializing the NARMAX model and generating its output from the passed data.'''

  StartIdx = max( Model.get_MaxNegOutputLag(), Model.get_MaxNegInputLag() ) # essentially q = max(qx, qy) as usual

  Model.set_OutputStorage( y[ : Model.get_MaxNegOutputLag() ].clone() ) # set previous y[k-j] states
  Model.set_InputStorage( tor.vstack( [ input[ : Model.get_MaxNegInputLag() ] for input in Data ] ) ) # set previous phi[k-j] states

  yHat = tor.zeros_like( y )
  yHat[ :StartIdx ] = y[ : StartIdx ].clone() # take solution samples, where Model hasn't got all data. Avoids init-Error spikes
  yHat[ StartIdx: ] = Model.Oscillate( [ input[ StartIdx: ] for input in Data ] )
  
  return ( yHat )
```

This function guarantees that given a set of input sequences (`Data`), the symbolic oscillator generates the ***EXACT*** same output sequence `y` as the original system (if both equations are the same). This requires particular care, since, per default, the SymbOsc is a zero-initialized "blank system", meaning that all initial conditions are zero.  
To illustrate, be the system $y[k] = a \cdot x[k] + b \cdot x[k-1] + c \cdot y[k-2]$. At time point $k=0$, the two last regressors ($y[k-2]$ and $x[k-1]$) are outside of the current buffer / sequence (negative indices) and are thus assumed to be zero if no other information is available. Those zeros (called initial conditions: $x[-1]$ and $y[-2]$, $y[-1]$) are taken from the object's internal buffers. However, if the "real" system that generated the output sequence $\underline y$ was already "running" before the sequence start, assuming that $y[-1] = 0$, $y[-2] = 0$ and $x[-1]=0$ could result in a completely different output sequence $\hat{\underline y}$. This problem is especially pronounced with  
a) NARMAX systems having a strong recursive component (strongly AR = auto-regressive) since the errors from the initial conditions are fed back into the system for a potentially long time.  
b) Chaotic or barely stable NARMAX systems, which can be "knocked-off" from their equilibrium state or just start in a completely different part of their phase-plot.  
Thus, it's very important to initialize the SymbOsc correctly, especially when comparing a SymbOsc output to a measurement, for example in cost functions or model validation procedures.

One can emulate a correct initialization from any arbitrary buffer by essentially using the buffer-start to fake the initial conditions, as shown in the above `InitAndComputeBuffer` function.

First, one needs to determine for how long one needs to "fake" having the initial conditions. This is simply the maximum of the system's negative input and output lags ($q_y$ and $q_x$), as obtained by the respective member functions `get_MaxNegInputLag` and `get_MaxNegOutputLag`:

```python
StartIdx = max( Model.get_MaxNegOutputLag(), Model.get_MaxNegInputLag() ) # essentially q = max(qx, qy) as usual
```

Then, the first $q_y$ samples (largest negative output lag: largest $j$ in any regressor containing a $y[k-j]$ in the NARMAX system) are taken from the measured system output $\underline y$ and used to overwrite the internal output buffer. The same is done with the set of input sequences $\underline x$. Now the SymbOsc is no longer a "blank" (zero-initialized) system but has valid (in the sense of real input and output data) initial conditions:

```python
Model.set_OutputStorage( y[ : Model.get_MaxNegOutputLag() ].clone() ) # set previous y[k-j] states
Model.set_InputStorage( tor.vstack( [ input[ : Model.get_MaxNegInputLag() ] for input in Data ] ) ) # set previous phi[k-j] states
```
Then, the SymbOsc output must copy the first $StartIdx$ measured output samples $\underline y$ into the output buffer, since those aren't calculated, being the initial conditions. From there on, the SymbOsc can normally be applied, since its first output samples are correct (have just been copied) and the internal buffers contain the corresponding initial conditions:

```python
yHat[ :StartIdx ] = y[ : StartIdx ].clone() # take solution samples, where Model hasn't got all data. Avoids init-Error spikes
yHat[ StartIdx: ] = Model.Oscillate( [ input[ StartIdx: ] for input in Data ] )
return ( yHat )
```

**Note:** For analysis purposes, the `NARMAX.SymbolicOscillator` object's internal buffers can be obtained at any time via the member functions `get_OutputStorage` and `get_InputStorage` and overwritten via `set_OutputStorage` and `set_InputStorage`. See next section for a complete list of member functions.

### 5.2 List of Member Functions

The following list shows all public member functions of the `NARMAX.SymbolicOscillator` object.

- **set/get_theta:** Getter and setter for the regression coefficients
- **set/get_OutputStorage:** Getter and setter for the internal output buffer
- **set/get_InputStorage:** Getter and setter for the internal input buffer
- **zeroInternalStorage:** Zeros the internal buffers, such that the system isn't influenced by previous buffer's data ("blank" system)
- **get_nRegressors:** Returns the number of regressors in the NARMAX expression the object represents
- **get_MaxNegInputLag:** Returns the largest negative input lag of the NARMAX expression the object represents
- **get_MaxPosInputLag:** Returns the largest positive input lag of the NARMAX expression the object represents. Positive lags allow to represent non-causal systems. It's weird and there are obviously problems at the buffer end (data is needed that doesn't currently exist), but people do what they want, really.
- **get_MaxNegOutputLag:** Returns the largest negative output lag of the NARMAX expression the object represents. There is no positive output lag, since I really can't predict the future outputs.
- **oscillate:** Processes a list of input sequences, and potentially noise and returns a list of output sequences, as described above.


[1] There are currently some limitations on the expression, such as nesting multiple functions with coefficients. This is, however, what I'm currently working on, since the Symbolic Oscillator is the object of the paper I'm writing at the moment. Support for arbitrary nesting depth of expressions will thus be added in the near future.

[Previous Tutorial: 1. Linear in the Parameters](https://github.com/Stee-T/NARMAX/tree/main/Examples/1_Linear_in_the_Parameters)  
[Next Tutorial: 2. Rational Fitting](https://github.com/Stee-T/NARMAX/tree/main/Examples/3_Rational_Fitting)