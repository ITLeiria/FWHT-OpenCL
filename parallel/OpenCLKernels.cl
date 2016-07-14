
#include "def.h"

#ifdef FOR_OPENCL_GPU

   __kernel void kStep0_PrepareMemory(__global DATATYPE* input, __global DATATYPE* output){
      ;//NOT NEEDED, dummy kernel
   }

   __kernel void kStep1_Had(__global DATATYPE* input, __global DATATYPE* output){
      ///////////////////////////
      //  Compute Definitions  //
      ///////////////////////////
      const int tid = get_local_id(0);
      const int numPositions = MATRIXperBLOCK*N;
      const int matrixOffSet = (tid/THREADSperMATRIX)*N;
      const int base = (get_group_id(0) * numPositions) ;//+ matrixOffSet;

      local DATATYPE sdata[MATRIXperBLOCK*N];
      __local DATATYPE *psdata = sdata + matrixOffSet;

      //////////////////////////////
      //  Copy From Global2Local  //
      //////////////////////////////
      //#pragma unroll
      for (int i = tid; i < MATRIXperBLOCK*N; i += BLOCKSIZE){
         sdata[i] = input[base + i];
      }

      //////////////////////////////////////////
      //  Hadamard even-radix-4 Calculations  //
      //////////////////////////////////////////
      const int pos = tid % THREADSperMATRIX;
      //#pragma unroll
      for (int stride = N >> 2; stride > 0; stride >>= 2)// if N=64 then stride=16
      {
         int lo = pos & (stride - 1);
         int i0 = ((pos - lo) << 2) + lo;
         int i1 = i0 + stride;
         int i2 = i1 + stride;
         int i3 = i2 + stride;

         barrier(CLK_LOCAL_MEM_FENCE);// needed!
         DATATYPE D0 = psdata[i0];
         DATATYPE D1 = psdata[i1];
         DATATYPE D2 = psdata[i2];
         DATATYPE D3 = psdata[i3];

         DATATYPE T;
         T = D0;
         D0         = D0 + D2;
         D2         = T  - D2;

         T = D1;
         D1         = D1 + D3;
         D3         = T  - D3;

         T = D0;
         psdata[i0] = D0 + D1;
         psdata[i1] = T  - D1;
         T = D2;
         psdata[i2] = D2 + D3;
         psdata[i3] = T  - D3;
      }


      ////////////////////////////////////////////////////////
      //  Final Hadamard even-radix-2 for odd power of two  //
      ////////////////////////////////////////////////////////
      if (log2N & 1) {
         barrier(CLK_LOCAL_MEM_FENCE);// just in case!

         for (int p = pos; p < N / 2; p += BLOCKSIZE/*get_local_size()*/){
            int i0 = p << 1;
            int i1 = i0 + 1;

            DATATYPE D0 = psdata[i0];
            DATATYPE D1 = psdata[i1];
            psdata[i0] = D0 + D1;
            psdata[i1] = D0 - D1;
         }
      }

      //////////////////////
      //  Reduce Results  //
      //////////////////////
      barrier(CLK_LOCAL_MEM_FENCE);// just in case!
      psdata[pos] = 
         ABS(psdata[pos])
         + ABS(psdata[pos+THREADSperMATRIX]);
      psdata[pos+THREADSperMATRIX] = 
         ABS(psdata[pos+2*THREADSperMATRIX])
         + ABS(psdata[pos+3*THREADSperMATRIX]);

      // 32 values reduction for each 16 threads
      for (unsigned int s=THREADSperMATRIX; s>0; s>>=1) {
         barrier(CLK_LOCAL_MEM_FENCE);// just in case!
         psdata[pos] += ABS(psdata[pos + s]);
      }

      ////////////////////
      //  Save Results  //
      ////////////////////
      if (pos == 0){
         output[ (get_group_id(0)*MATRIXperBLOCK)+(tid/THREADSperMATRIX) ] = sdata[tid/THREADSperMATRIX*N];
      }
   }

#elif FOR_OPENCL_CPU

   __kernel void kStep0_PrepareMemory(__global DATATYPE* pidata, __global DATATYPE* podata){
      const unsigned int pos = get_global_id(0);
      const unsigned int matrixOffset = pos*N*MATRIXperBLOCK;

      pidata += matrixOffset;
      podata += matrixOffset;

      for (int i = 0; i < N; i++){
         for (int m = 0; m < MATRIXperBLOCK; ++m){
            podata[i*MATRIXperBLOCK+m] = pidata[i+m*N];
         }
      }
   }  


   __kernel void kStep1_Had(__global DATATYPEV* psdata, __global DATATYPEV* output){
      const unsigned int pos = get_global_id(0);
      const unsigned int matrixOffset = pos*N;

      psdata += matrixOffset;

      /////////////////////////////
      //  Hadamard even-radix-2  //
      /////////////////////////////
      for (int stride = N / 2; stride >= 1; stride >>= 1){//stages with different butterfly strides
         for (int base = 0; base < N; base += 2 * stride){//subvectors of (2 * stride) elements
            for (int j = 0; j < stride; j++){//butterfly index
               int i0 = base + j +      0;
               int i1 = base + j + stride;

               DATATYPEV T1 = psdata[i0];
               DATATYPEV T2 = psdata[i1];
               psdata[i0] = T1 + T2;
               psdata[i1] = T1 - T2;
            }
         }
      }

      //////////////////////
      //  Reduce Results  //
      //////////////////////
      __private DATATYPEV sumV = ABS(psdata[0]);

      for (int i = 1; i < N; i++) {
         sumV += ABS(psdata[i]);
      }

      ////////////////////
      //  Save Results  //
      ////////////////////
      output[pos] = sumV;
   }

#endif
