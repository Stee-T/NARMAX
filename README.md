# FOrLSR Papers supporting code

I'll soon submit for peer review to IEEE my research on NARMAX systems, where I propose an optimized version of the **FOrLSR** (*forward orthogonal Least squares regression*) in addition to two new algorithmic classes in that framework (each in its own paper). 
In short, the FOrLSR algorithms allow to obtain an analytical expression (aka symbolic representation) for any unknown system or function. The abstract and the conclusion of the papers give an overview of the work. Note that the critical parts of the papers (pseudo-codes, theorem, proofs, etc) have been removed, as the work is currently unpublished and thus not yet registered as my intellectual property.

## Paper 1: AOrLSR - Arborescent Orthogonal Least Squares Regression

The first part (rFOrLSR) presents a common algorithm but with its nested loops being transformed in a single matrix operation (for heavy parallelization and GPU operation) and made recursive to flatten the complexity from quadratic to linear. → linear algebra-based optimization with numerics.

The second part of the paper is the arborescence design, which is essentially a linear algebra and graph-theory-based optimization procedure to tackle the NP-hard problem of finding the linear equation solution with the largest amount of zero entries in the solution vector and the smallest solving error.
Read or download the AOrLSR paper.

## Paper 2: DMOrLSR - Dictionary Morphing Orthogonal Least Squares Regression

The second paper morphs the user-passed regressor (functions) to allow expansions which adapt their terms to the system output (imagine a fourier transform which finds the exact peaks and then only contains those terms, being sparse). This is based on genetic algorithms, linear algebra, matrix calculus / infinitesimal optimization, with closed form gradients and Hessians independent of the user-passed function, number of arguments and data.
Read or download the DMOrLSR paper.

## Python Package for both Papers:
Those algorithms will soon be released here as a GPU-accelerated python machine learning package.

The library makes optimal sparse least squares fitting of any vectors passed by the user for the given system response. All provided examples are with severely non-linear auto-regressive systems, however the user can pass dictionaries with any type of functions. This is thus a very general mathematical framework supporting for example wavelet-, RBF, polynomial, etc fitting. The papers also contain an example of using the library to fit a nested expansion in |x|^n inside a fraction or of nesting non-linearities inside other non-linearities. 
