
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "def.h"
#include "OpenCLModule.h"


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

    gtOpenclInitialize();

    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
    h_Data      = (DATATYPE *)malloc(DATASIZE);
    printf("...allocating GPU memory\n");
    gtOpenclInitMem(MemInput,  sizeof(DATATYPE), DATASIZE);
    gtOpenclInitMem(MemPrepare, sizeof(DATATYPE), DATASIZE);

    printf("...generating data\n");
    printf("Data length: %i\n", dataN);
    srand(2007);

    for (i = 0; i < dataN; i++)
    {
        h_Data[i] = (DATATYPE)rand() / (DATATYPE)RAND_MAX;
    }

    printf("Running OpenCL-%s Fast Walsh Transform...\n", (USE_GPU==1)?"GPU":"CPU");
    #if USE_GPU == 1
        gtOpenclCopyH2D(MemInput, h_Data, sizeof(DATATYPE), 0, dataN);
        gtOpenclSetKernelGlobalSize(0, k1Had, (numMatrixs/MATRIXperBLOCK)*BLOCKSIZE);
        gtOpenclSetKernelLocalSize (0, k1Had, BLOCKSIZE);

        printf("running kernel   for: <<%d, %d>>\n",  (int)*gtOpenclGetKernelGlobalSize(0,k1Had), (int)*gtOpenclGetKernelLocalSize(0,k1Had) );
        printf("                 for: %d matrix, size %dx%d\n",  numMatrixs, 1<<(log2N/2), 1<<(log2N/2) );
        printf("            needding: %d threads/matrix\n", THREADSperMATRIX);
        printf("               doing: %d matrix/warp\n", MATRIXperWARP);
        printf("               doing: %d matrix/block\n", MATRIXperBLOCK);
        printf("      max calculated: %d matrix\n", MATRIXperBLOCK*(numMatrixs/MATRIXperBLOCK));
        int smem = MATRIXperBLOCK*N*sizeof(DATATYPE);
        printf("            needding: %d smem/block\n", smem);
    #else
        if(gtOpenclGetVectorWidth(0)>1){
            printf("Rearranging input buffer to align with CPU SIMD vetor intructions of size %d...\n", gtOpenclGetVectorWidth(0));
            gtOpenclCopyH2D(MemPrepare, h_Data, sizeof(DATATYPE), 0, dataN);
            gtOpenclSetKernelGlobalSize(0, k0Prepare, numMatrixs/gtOpenclGetVectorWidth(0));
            gtOpenclSetKernelLocalSize (0, k0Prepare, BLOCKSIZE);
            printf("__Launching PrepareBuffer kernel__\n");
            gtOpenclLaunchKernel0();
        }else{
            gtOpenclCopyH2D(MemInput, h_Data, sizeof(DATATYPE), 0, dataN);
        }

        gtOpenclSetKernelGlobalSize(0, k1Had, (numMatrixs/gtOpenclGetVectorWidth(0))*BLOCKSIZE);
        gtOpenclSetKernelLocalSize (0, k1Had, BLOCKSIZE);

        printf("running kernel   for: <<%d, %d>>\n",  (int)*gtOpenclGetKernelGlobalSize(0,k1Had), (int)*gtOpenclGetKernelLocalSize(0,k1Had) );
        printf("                 for: %d matrix, size %dx%d\n",  numMatrixs, 1<<(log2N/2), 1<<(log2N/2) );
        printf("            needding: %d threads/matrix\n", THREADSperMATRIX);
        printf("               doing: %d matrix/warp\n", MATRIXperWARP);
        printf("               doing: %d matrix/block\n", gtOpenclGetVectorWidth(0));
        printf("      max calculated: %d matrix\n", gtOpenclGetVectorWidth(0)*(numMatrixs/gtOpenclGetVectorWidth(0)));
    #endif

    printf("__Launching HAD kernel__\n");
    timestamp_t cputime = get_timestamp();
    gtOpenclLaunchKernel1();
    cputime = (get_timestamp() - cputime);
    printf("           run time : %.7Lf\n",cputime/1000000.0L);

/*
    printf("Reading back results...\n");
    gtOpenclCopyD2H(MemInput, h_Data, sizeof(DATATYPE), 0, numMatrixs);
    for (int i = 0; i < numMatrixs; ++i){
        #if USE_FLOAT == 1
            printf("sum[%02d]=%.0f\n",i,h_Data[i]);
        #else
            printf("sum[%02d]=%d\n",i,h_Data[i]);
        #endif
    }printf("\n");
*/

    printf("Shutting down...\n");
    free(h_Data);

    gtOpenclRelease();
    fflush(stdout);

    return 0;
}
