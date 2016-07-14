# FWHT-Sequential
A normal Fast Walsh-Hadamard Transform (FWHT) implementation (with matrix sum reduction at the end).

### Definitions
* File [def.h] stores the program-wide definition of the number of matrices to process in runtime (_numMatrixs_), and the _log2_ size of each individualy matrix composed of _N_ elements of datatype _DATATYPE_.

* File [hadocl.c] contains the program definition as well as the FWHT function.


[def.h]: def.h
[hadocl.c]: hadocl.c
