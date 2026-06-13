<div align="justify">

# Example 0: Minimal example and intuition

This example / tutorial illustrates
- The general use of the library
- The content
- Minimal intuitive example

## 0. Toy example to illustrate the library
Let's start with a toy example to illustrate the idea behind the library.
Imagine you have a serious gambling addiciton and thus you bet for an insane amount of money with your friend that if he thinks of a random multivariate polynomial you can always find it. Since there are an infinite number of such polynomials, that's a slighlty unfair bet, so you agree on the following constraints:

The variables are limited to $x$, $y$, $z$, the polynomial is made of any powers below 5 (thus $x$, $y$ , $z$, $x^2$, $y^2$, $z^2$, $x^3$, ... $x^{5}$, $y^{5}$, $z^{5}$) with any decimal coefficients in $\left[-1,1\right]$ with increments of 0.1 (thus -1, -0.9, ... 0.9, 1). In addition, you negociate that you are allowed to see the polynomial evaluated on one instance of random data.
Now, the problem is limited to a mere "choose 5 of 15 * 21 = 315" which is $315C5=2.5\cdot10^{10}$ possible guesses. Thus, a correct guess is about 200 times less likely than winning the Jackpot of EuroMillions (in 2025).


Considering the amount of money involved in your bet, you want to make sure that we can always win this game! Here are the steps:

First we generate observations of our variables $x$, $y$ and $z$ from a uniform distribution, which are 1D torch tensors.

```python
import numpy as np
import torch as tor
import NARMAX # of course!

# A) Generate the friend's data
tor.manual_seed( 42 ) # for reproducibility
NumSamples = 250
x, y, z = [ tor.rand( NumSamples, 1 ) for _ in range( 3 ) ] # 1D tor.Tensor
```

Then, using the generated variable observations ($x$, $y$, $z$), we generate all possible expressions (monomials) that are valid candidates for our friend's polynomial. We agreed on having a maximal degree of 5 and thus we loop to generate all monomials up to that degree as tensors and strings (the library needs both).

The Regressornames list will be:
```python
['x', 'y', 'z', 'x^2', 'y^2', 'z^2', 'x^3', 'y^3', 'z^3', 'x^4', 'y^4', 'z^4', 'x^5', 'y^5', 'z^5']
```
The code:

```python
# B) Generate all possible terms
MaxDegree: int = 5

AllMonomials: list[ tor.Tensor ] = []
RegressorNames: list[ str ] = []

for var, name in zip( [ x, y, z ], [ "x", "y", "z" ] ):
  for power in range(1, MaxDegree + 1):
    AllMonomials.append( var ** power )
    RegressorNames.append( f"{ name }^{ power }" )
```
Finally, we simulate getting an evaluated polynomial from our friend. Note that importantly, the resulting tor tesor doesn't contain any strings or auto-grad information about which terms were used to design the polynomial.

```python
# C) Simulate our friend randomly designing a polynomial (hardcoded to keep the example minimal)
Evaluation: tor.Tensor = 0.3*x - 0.8*y**4 + 0.4*z**3 - 0.5*x**3 + 0.9*y**2
```

We can now use the library. The required information is the "system output", here the evaluated polynomial in a torch tensor.

```python
# D) Use the library!
Arbo = NARMAX.Arborescence( Evaluation, # 1D tor.Tensor: Our friend's polynomial
                            Dc = tor.column_stack( AllMonomials ), # 2D tor.Tensor: Dictionary of Candidates (Dc)
                            DcNames = np.array( RegressorNames ), # NDArray: Names of Dc's columns
                          )

theta, L = Arbo.fit()[:2] # Coefficients and indices

print( "\nRecognized regressors:" )
for i in range( len( L ) ): print( theta[i].item(), RegressorNames[ L[i] ] )
```



The output is correct up to floating point rounding errors, we are now very rich!

```
Recognized regressors:
0.3 x^1
-0.49999999999999994 x^3
0.8999999999999999 y^2
-0.7999999999999999 y^4
0.4 z^3
```

***On a more serious note:*** This illustrates the basic idea behind this library: We have an unknown system / function that we want to describe with known functions (in this case retrieving a multivariate polynomial). This yields human-readable equations which can be used to learn about physical phenomena (reaction of some material to applied forces or chemicals, etc.) or used to control systems.

Imagine having a complex robotic system where you need to control the movement of certain parts. The responce of those parts will depend on many factors such as their combined weight, the strengths of their motors, the current position of other parts and if the robot is already in movement, etc. To accurately control the robot, one can thus either do

a) very complex computer simulations where all of those parameters are hard-coded whith their exact respective equations.

b) Fit a neural network or similar: great fitting power but still a blackbox and not thus doesn't generate verifiably correct representation.

c) A NARMAX, which gives us the best of both words: interpretable and verifiably correct equations with great fitting power.


[Next Tutorial: 1. Linear in the parameters System](https://github.com/Stee-T/NARMAX/tree/main/Examples/1_Linear_in_the_Parameters)  