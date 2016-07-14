# FWHT-OpenCL
A OpenCL-C Fast Walsh-Hadamard Transform (FWHT) implementation (with matrix sum reduction at the end) optimized for both GPU and CPU.

### Definitions
* File [def.h] stores the program-wide definition of the number of matrices to process in runtime (_numMatrixs_), and the _log2_ size of each individualy matrix composed of _N_ elements of datatype _DATATYPE_. Additionally, flag _USE_GPU_ controls if the implementation will compile for a GPU environment or a CPU environment.

* File [OpenCLModule.c] contains all OpenCL management related code. The OpenCL profiling routines are located here. As a final detail, if the CPU environment is selected, the algorithm will retrive the CPU SIMD preferred lane width to further apply that size in the OpenCL-CPU kernel datatypes at [OpenCLKernels.cl].

* File [OpenCLKernels.cl] contains both GPU and CPU optimal kernel definitions of the FWHT. While the GPU implementation divides matrix transformations by a given number of GPU-threads (_THREADSperMATRIX_), in the CPU implementation each matrix is processed by a single CPU-core.

* File [hadocl.c] contains the main program definition.

[def.h]: def.h
[OpenCLModule.c]: OpenCLModule.c
[OpenCLKernels.cl]: OpenCLKernels.cl
[hadocl.c]: hadocl.c
