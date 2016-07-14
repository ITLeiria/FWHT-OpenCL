
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "def.h"


////////////////////////////////////////////////////////////////////////////////
// CPU Sequential Walsh-Hadamard Transform
////////////////////////////////////////////////////////////////////////////////
void cpuHad(DATATYPE* input, DATATYPE* output){
    for (unsigned int pos = 0; pos < numMatrixs; ++pos) {
        unsigned int matrixOffset = pos*N;

        DATATYPE *psdata = input + matrixOffset;

        /////////////////////////////
        //  Hadamard even-radix-2  //
        /////////////////////////////
        for (int stride = N / 2; stride >= 1; stride >>= 1){//stages with different butterfly strides
            for (int base = 0; base < N; base += 2 * stride){//subvectors of (2 * stride) elements
                for (int j = 0; j < stride; j++){//butterfly index
                    int i0 = base + j +      0;
                    int i1 = base + j + stride;

                    DATATYPE T1 = psdata[i0];
                    DATATYPE T2 = psdata[i1];
                    psdata[i0] = T1 + T2;
                    psdata[i1] = T1 - T2;
                }
            }
        }

        //////////////////////
        //  Reduce Results  //
        //////////////////////
        DATATYPE sumV = fabs( psdata[0] );

        for (int i = 1; i < N; i++) {
            sumV += fabs(psdata[i]);
        }

        ////////////////////
        //  Save Results  //
        ////////////////////
        output[pos] = sumV;
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU timestamp helper
////////////////////////////////////////////////////////////////////////////////
#include <sys/time.h>
#include <stdio.h>

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000LL;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
#define dataN  ((1 << log2N)*numMatrixs)
const unsigned int DATASIZE = dataN * sizeof(DATATYPE);

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    DATATYPE *h_Data;

    int i;
    printf("%s Starting...\n\n", argv[0]);

    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
    h_Data      = (float *)malloc(DATASIZE);

    printf("...generating data\n");
    printf("Data length: %i\n", dataN);
    srand(2007);

    for (i = 0; i < dataN; i++)
    {
        h_Data[i] = (DATATYPE)rand() / (DATATYPE)RAND_MAX;
    }

    printf("Running CPU Fast Walsh Transform...\n");
    printf("running function for: %d matrix, size %dx%d\n",  numMatrixs, 1<<(log2N/2), 1<<(log2N/2) );
    
    timestamp_t cputime = get_timestamp();
    cpuHad(h_Data, h_Data);
    cputime = (get_timestamp() - cputime);
    printf("           CPU time : %.7Lf\n",cputime/1000000.0L);

/*
    printf("Reading back results...\n");
    for (int i = 0; i < numMatrixs; ++i){
        printf("sum[%02d]=%.0f\n",i,h_Data[i]);
    }printf("\n");fflush(stdout);
*/

    printf("Shutting down...\n");
    free(h_Data);

    return 0;
}
