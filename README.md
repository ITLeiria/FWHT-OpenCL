# FWHT-OpenCL
Fast Walsh-Hadamard Transform (FWHT) implementation over OpenCL for GPU and CPU.

This project includes:
  - a [Sequential] FWHT implementation in C
  - a [parallel] FWHT implementation resolting to OpenCL C:
    - Optimized for manywarp GPU enviroments
    - Optimized for manycore CPU enviroments (with or without SIMD)


[Sequential]: <https://github.com/ITLeiria/FWHT-OpenCL/tree/master/sequential>
[parallel]: <https://github.com/ITLeiria/FWHT-OpenCL/tree/master/parallel>
