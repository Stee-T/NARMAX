<div align="justify">

Welcome to the example / tutorial section. Each example folder contains a python script with the fitting code and an explanation README file displayed as github page.

# Example 1: Basics & Linear-in-the-parameter system
Covers the basics of the library (functions, algorithm hyper-parameters, etc) and demonstrates how to use the most common signal-constructors and the arborescence.

**Fitted system**: $y\[k\] = 0.2x\[k\] + 0.3x^3\[k-1\] + 0.7|x\[k-2\]x^2\[k-1\]| +0.5e^{x\[k-3\]x\[k-2\]} - \cos(y\[k-1\]x\[k-2\]) -0.4|x\[k-1\]y^2\[k-2\]| - y^3\[k-3\]$

# Example 2: The Symbolic Oscillator
Showcases the Symbolic Oscillator object `NARMAX.SymbolicOscillator`, which allows to execute the NARMAX System (user-defined or as obtained by the arborescence object) and thus apply it on data.
In addition, the SymbolicOscillator object supports modulating the regression parameters and has a supplementary input allowing experimentation with additional regressors, DC-offset, additive noise, etc.

**Executed system**: $y\[k\] = W\[k\] + \frac{{\theta_0\frac{y\[k-1\]}{x_2\[k\]} + \theta_1 x_1^2\[k-1\] + \frac{\theta_2}{\text{abs}\left( 0.2x_1\[k-1\] + 0.5x_1\[k-2\]x_2\[k\] - 0.2 \right)} }}{1 + \theta_3 x_1\[k-1\]x_2\[k-1\] + \theta_4 x_2^2\[k-2\] + \theta_5\cos\left( 0.2x_1\[k-3\]x_2\[k-1\] - 0.1 \right) }$  
This system has two inputs $x_1$ and $x_2$ in addition to an internal noise channel $W[k]$ and a single output (MISO system).

# Example 3: Rational System
Demonstrates how to fit rational models with the provided signal-constructor. It also contains some supplementary information about the arborescence fitting, including how to reliably reproduce results.

**Fitted system**: $y\[k\]=\frac{0.6|x\[k\]|-0.35x^3\[k\]-0.3x\[k-1\]y\[k-2\]+0.1|y\[k-1\]|}{1-0.4|x\[k\]|+0.3|x\[k-1\]x\[k\]|-0.2x^3\[k-1\]+ 0.3y\[k-1\]x\[k-2\]}$

# Example 4: Imposing Regressors
Demonstrates how to use the imposed regressor dictionary. This example illustrates how to fit a (linear) IIR filter of a desired order.

**Fitted system**: 5 parallel Biquadratic IIR filters (5 parallel Second Order Sections) fitted with a single IIR filter (ARX system).


# Example 5: Custom Expansion And Validation
Demonstrates how to generate embedded expansions (advanced linearization and order determination) and how to create a custom dictionary and validation function.

**Fitted system**: $y = \text{sgn}(x)(1-\frac{1}{1+|x|A})$ with $A≔\Sigma_{j\in J}\theta _j |x|^j$ and $J\subseteq \mathbb{N}$.


# Example 6: Multiple Input Multiple Output System
Demonstrates how to fit MIMO (Multiple Input Multiple Output) systems.

**Fitted system**:  
$y_1\[k\] = 0.2 x_1\[k\] + 0.3 x_2^3\[k\] + 0.7 |x_3\[k\]| + 0.5 x_2\[k-3\] x_1\[k-2\] - 0.3 y_2\[k-1\] x_2^2\[k-2\] - 0.8 |x_3\[k-1\] y_1\[k-2\]| - 0.7 x_1\[k-1\] x_2^2\[k-1\]$  
$y_2\[k\] = 0.3 x_1\[k-1\] + 0.5 x_3^3\[k\] + 0.7 |y_1\[k-1\]| + 0.6 y_1\[k-3\] x_1\[k-2\] - 0.4 y_1\[k-1\] x_3^2\[k-2\] - 0.9 |x_3\[k-1\] y_2\[k-2\]| - 0.7 x_3\[k-1\] x_2^2\[k-1\]$

# Example 7: Binary System Fitting
Demonstrates how to fit binary systems and recapitulates the previous tutorials with an example from the paper: "Booler"-Ctor, MISO fitting & custom validation functions. It shows that very low error can be obtained, even if the retrieved equation is not the exact one of the system but an equivalent expression.

**Fitted system**: $y[k] = ( !x1\[k\] \&\& x2\[k\] ) - ( x3\[k\] \lor x4\[k\] ) + ( x1\[k\] \|\| x3\[k-1\] ) + ( x2\[k\] \lor x4\[k-2\] ) - ( !x1\[k-2\] \&\& x2\[k\] ) - !x3\[k-2\] + !x2\[k-2\] + x4\[k-1\]$