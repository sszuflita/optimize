# optimize
A dead simple cuda black-box optimization library created for CS 179 at Caltech.

## background

Black box optimization involves numerically optimizing a function which has no assumed structure. This lack of structure means that more advanced numerical optimization techniques (gradient descent, Lagrangian analysis, primal optimization, etc.) may not be appropriate.

This type of problem is very well suited for a GPU, in which many function evaluations can be computed in parallel. Combining these subproblems (choosing the maximum) is fairly cheap, and doesn't require much synchronization. There also isn't much data that needs to pass back and forth between the GPU and CPU, meaning that this problem isn't IO-bottlenecked.

## approach

This library presents a dead-simple interface. The user defines a __device__ function, which is run by the GPU a fixed number of times (the number of function evaluations is a key metric in black box optimization).

The goal of this project was to produce a simple example which could be tested and benchmarked, so the current input space is a line segment. This could fairly easily be expanded to multiple dimensions, although a more generic input space would be difficult to efficiently describe.

This library was also designed to support multiple optimization algorithms. Currently, the only implemented method is a very simple Monte Carlo method which uniformly samples the input space. This approach was chosen because it should work well for a diverse range of functions.

Admittedly, the interface is also a bit awkward; ideally the user's objective function should exist in their code, and it should be passed into the optimize call. Unfortunately, [cuda doesn't offer good support for passing host functions onto device](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-pointers).

## results

## references
