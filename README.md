# The optimization algorithm for minimizing non-negative quadratic problems under cardinality constraint.

This code solves the following problem:

`minimize_x 0.5*x'*H*x + C'*x`

  `                   s.t   x>=0  , ||x||_0<T0`

The detail of the algorithm is described in the following paper:
'Confident kernel dictionary learning for discriminative representation
of multivariate time-series', B. Hosseini, F. Petitjean, Forestier G., and B. Hammer.

## Using NQP
- Try NQP_demo.m for a simple example for the proper usage of the algorithm.
