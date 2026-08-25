# Mathematics Reference Notes

## Derivatives and the Chain Rule
For a composite function f(g(x)), the chain rule gives the derivative as
f'(g(x)) * g'(x). This underlies backpropagation in neural networks, where the
gradient of the loss with respect to an early layer's weights is computed by
multiplying local gradients along the computation graph.

## Probability Basics
For independent events A and B, P(A and B) = P(A) * P(B). Bayes' theorem
relates conditional probabilities: P(A|B) = P(B|A) * P(A) / P(B). Expected
value is the probability-weighted average of a random variable's possible
outcomes.

## Matrix Determinants
The determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc. A matrix is
invertible if and only if its determinant is nonzero. For larger matrices,
the determinant can be computed by cofactor expansion or, more efficiently,
via LU decomposition.

## Least-Squares Regression
Ordinary least squares finds the parameter vector that minimizes the sum of
squared residuals between predicted and observed values. For a linear model
y = X*beta, the closed-form solution is beta = (X^T X)^-1 X^T y, provided
X^T X is invertible.

## Big-O Notation
Big-O notation describes an upper bound on growth rate as input size grows
without bound, ignoring constant factors and lower-order terms. O(n) grows
linearly, O(log n) grows logarithmically, and O(2^n) grows exponentially.
