# optimize
A dead simple cuda black-box optimization library created for CS 179 at Caltech.

## background

Black box optimization involves numerically optimizing a function which has no assumed structure. This lack of structure means that more advanced numerical optimization techniques (gradient descent, Lagrangian analysis, primal optimization, etc.) may not be appropriate.

This type of problem is very well suited for a GPU, in which many function evaluations can be computed in parallel. Combining these subproblems (choosing the maximum) is fairly cheap, and doesn't require much synchronization. There also isn't much data that needs to pass back and forth between the GPU and CPU, meaning that this problem isn't IO-bottlenecked.

## approach

This library presents a dead-simple interface. The user defines a \_\_device\_\_ function, which is run by the GPU a fixed number of times (the number of function evaluations is a key metric in black box optimization).

The goal of this project was to produce a simple example which could be tested and benchmarked, so the current input space is a line segment. This could fairly easily be expanded to multiple dimensions, although a more generic input space would be difficult to efficiently describe.

This library was also designed to support multiple optimization algorithms. Currently, the only implemented method is a very simple Monte Carlo method which uniformly samples the input space. This approach was chosen because it should work well for a diverse range of functions.

Admittedly, the interface is also a bit awkward; ideally the user's objective function should exist in their code, and it should be passed into the optimize call. Unfortunately, [cuda doesn't offer good support for passing host functions onto device](http://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-pointers).

## results

I began by benchmarking the GPU optimizer against a simple CPU optimizer using an analagous serial algorithm. I used the objective function -(x - 500)^2.

![](https://github.com/sszuflita/optimize/blob/master/analysis/-(x-500)%5E2_full.png?raw=true "Optional title")

We can see that for 1e8 points (the largest N), the GPU really dominates (2724.387939 ms vs 98.551132 ms). Stripping out these points, we can see that the two implementations are similar for smaller points.

![](https://github.com/sszuflita/optimize/blob/master/analysis/-(x-500)%5E2_stripped.png?raw=true "Optional title")

As a sanity check, we want to make sure the output maximums are actually good estimations of the maximum. I threw out the N=10 and N=100 cases, but for the rest we can see that both the CPU and GPU converge on the maximum fairly accurately.

![](https://github.com/sszuflita/optimize/blob/master/analysis/accuracy.png?raw=true "Optional title")

One further question we may have is how the GPU performs with more complex functions. As mentioned in class, GPUs have special hardware to compute transcendental functions. To test the performance of this, I reran the benchmarks with

f(x) = 20 * log(x) - .0004 * x * x * sin(x) + x * cos(x)

![](https://github.com/sszuflita/optimize/blob/master/analysis/complex.png?raw=true "Optional title")

We can see that the GPU still beats the CPU handily for larger numbers of input values.

## references
I looked at some of the slides, that's about it.
