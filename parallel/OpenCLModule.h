#ifndef _OPENCLMODULE_H_
#define _OPENCLMODULE_H_

#include <stdio.h>
#include <CL/cl.h>
#include "def.h"

#ifdef __cplusplus
extern "C"{
#endif

    //////////////////////
    //  OpenCL Kernels  //
    //////////////////////
    typedef enum __attribute__ ((__packed__)) { 
      k0Prepare = 0,
      k1Had,
    Klength} enumKernel;



    //////////////////////
    //  OpenCL Buffers  //
    //////////////////////
    typedef enum __attribute__ ((__packed__)) { 
      MemInput = 0,
      MemPrepare,
    Mlength} enumMem;



    ////////////////////////
    //  OpenCL Functions  //
    ////////////////////////
    void gtOpenclInitialize();
    void gtOpenclRelease();

    void gtOpenclHandleError(int err, const char *file, int line, const char *fmt, ... );
    #define HANDLE_ERROR(err,...) (gtOpenclHandleError(err, __FILE__, __LINE__ , __VA_ARGS__))

    #ifdef DEBUGGTOPENCL
        typedef enum { cpy_h2d, cpy_d2h, cpy_d2d, maps, unmaps } enumMemTransferType;
        void gtOpenclDebugEventPreExecutionTimeK(cl_event event, int deviceIndex, enumKernel kernel);
        void gtOpenclDebugEventPreExecutionTimeM(cl_event event, int deviceIndex, enumMem mem, enumMemTransferType type);
        void gtOpenclDebugEventExecutionTimeK(cl_event event, int deviceIndex, enumKernel kernel);
        void gtOpenclDebugEventExecutionTimeM(cl_event event, int deviceIndex, enumMem mem, enumMemTransferType type);
    #endif

    size_t gtOpenclGetNumDevices();
    
    cl_device_id     *gtOpenclGetDevice  (int device);
    cl_context       *gtOpenclGetContext (int device);
    cl_command_queue *gtOpenclGetQueue   (int device);
    cl_program       *gtOpenclGetProgram (int device);

    cl_uint    gtOpenclGetWarpSize                (int device); 
    cl_uint    gtOpenclGetDeviceComputeCapability (int device); 
    cl_uint    gtOpenclGetMaxComputeUnits         (int device); 
    cl_ulong   gtOpenclGetMaxLocalMemory          (int device); 
    cl_uint    gtOpenclGetVectorWidth             (int device); 

    cl_kernel        *gtOpenclGetKernel  (int device, enumKernel index);
    size_t   *gtOpenclGetKernelGlobalSize(int device, enumKernel index);
    void      gtOpenclSetKernelGlobalSize(int device, enumKernel index, size_t size);
    size_t   *gtOpenclGetKernelLocalSize (int device, enumKernel index);
    void      gtOpenclSetKernelLocalSize (int device, enumKernel index, size_t size);

    cl_mem *gtOpenclGetMem(int device, enumMem index);
    void gtOpenclInitMem(enumMem memIndex, size_t sizeOfType, unsigned int elemLength);


    void gtOpenclLaunchKernel0();
    void gtOpenclLaunchKernel1();
    void gtOpenclCopyD2H(enumMem memIndex, void *buffer, size_t sizeofDataType, unsigned int elems_start, unsigned int elems_end);
    void gtOpenclCopyH2D(enumMem memIndex, void *buffer, size_t sizeofDataType, unsigned int elems_start, unsigned int elems_end);


#ifdef __cplusplus
}
#endif

#endif
