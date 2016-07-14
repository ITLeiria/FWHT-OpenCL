#ifndef _DEF_H_
#define _DEF_H_

#ifdef __cplusplus
extern "C"{
#endif

	#define USE_FLOAT 1
	#define USE_GPU 1
	

	#define log2N 10
	#define N (1<<log2N)
	#define numMatrixs ((9*9*9*9)*8)

	#if USE_FLOAT == 1
		typedef float DATATYPE;
	#else
		typedef int DATATYPE;
	#endif

	#if USE_GPU == 1
		#define WARPSIZE 256
		#define WARPSperBLOCK 1
		#define BLOCKSIZE (WARPSperBLOCK*WARPSIZE)
		#define THREADSperMATRIX (N/4)
		#define MATRIXperWARP (WARPSIZE/THREADSperMATRIX)
		#define MATRIXperBLOCK (MATRIXperWARP*WARPSperBLOCK) // or (BLOCKSIZE/THREADSperMATRIX)
	    // log2N=2   matrix2x2    N=4     used in HEVC   then need 1   threads per matrix   32 matrix per warp32   SMEM = N*sizeof(float)*32 =  4*4*32 = 512
	    // log2N=3   matrix4x2    N=8                    then need 2   threads per matrix   16 matrix per warp32
	    // log2N=4   matrix4x4    N=16    used in HEVC   then need 4   threads per matrix    8 matrix per warp32   SMEM = N*sizeof(float)*8  = 16*4*8  = 512
	    // log2N=5   matrix8x4    N=32                   then need 8   threads per matrix    4 matrix per warp32
	    // log2N=6   matrix8x8    N=64    used in HEVC   then need 16  threads per matrix    2 matrix per warp32   SMEM = N*sizeof(float)*2  = 64*4*2  = 512
	    // log2N=7   matrix16x8   N=128                  then need 32  threads per matrix    1 matrix per warp32
	    // log2N=8   matrix16x16  N=256                  then need 64  threads per matrix  1/2 matrix per warp32
	    // log2N=9   matrix32x16  N=512                  then need 128 threads per matrix
	    // log2N=10  matrix32x32  N=1024                 then need 254 threads per matrix
	#else
		#define WARPSIZE 1
		#define WARPSperBLOCK 1
		#define BLOCKSIZE 1
		#define THREADSperMATRIX 1
		#define MATRIXperWARP 1
	#endif


#ifdef __cplusplus
}
#endif

#endif
