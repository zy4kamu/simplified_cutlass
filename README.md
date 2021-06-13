# Simplified CUTLASS

The origin of the project was to understand how NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass) works in details. 
The project is amazing, well organized and very interesting.
But at the same time it is rather difficult to get through all this template magic, policies for different architectures and and tricky optimizations.
There was a desire to have some very basic but working implementation of BLAS on GPU which can be used as a starting point for new comers and people who 
just want to understand how it works under the hood.

As a result a very basic strategy described in https://github.com/NVIDIA/cutlass was implemented. 
Everything related to optimzations (pipelining, WMMA, different architectures), google testing, CMake configurations, C++ templates, etc. was roughly 
dropped out with the goal to have a minimally possible working example on hands.

![alt text](https://raw.githubusercontent.com/NVIDIA/cutlass/master/media/images/gemm-hierarchy-with-epilogue-no-labels.png)

## How To

In general everything is simple:
```
mkdir build && cd build && cmake .. && make
./examples/example
```
You should see `Passed` on the screen.
Tested on Ubuntu 18.04 with cmake 3.17.5 and CUDA 10.0 installed.

## Restrictions
* Only `float32`
* All matrix sized must be divisible by 128.
