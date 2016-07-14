#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdarg.h>
#include "OpenCLModule.h"


//////////////////////
//  OpenCL Kernels  //
//////////////////////
#ifdef DEBUGGTOPENCL
  static const char *enumKernelName[] = {
      "kStep0_PrepareMemory",
      "kStep1_Had",
  };

  static const char *enumKernelDisplayName[] = {
      "k0_Prepare",
      "k1_Had",
  };
#endif

//////////////////////
//  OpenCL Buffers  //
//////////////////////

static const cl_mem_flags enumMemFlags[] = {
	CL_MEM_READ_WRITE, CL_MEM_READ_WRITE, CL_MEM_READ_WRITE,
};

static const void *enumMemPointers[] = {
	NULL, NULL, NULL, 
};

#ifdef DEBUGGTOPENCL
  static const char *enumMemName[] = {
    "MemInput", "MemPrepare",
  };

  static const char *enumMemDisplayName[] = {
    "MemInput", "MemPrepare",
  };
#endif

///////////////////////////////////////
//  OpenCL All-In-One DataStructure  //
///////////////////////////////////////
#ifdef DEBUGGTOPENCL
	typedef struct __attribute__ ((__packed__)) oclDebugInfo {
		float time_queued_submited;
		int nr_queued_submited;
		float time_submited_running;
		int nr_submited_running;
		float time_running;
		int nr_running;
	} oclDebugInfo;
#endif

typedef struct __attribute__ ((__packed__)) oclDeviceData {
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	//---
	cl_uint warpSize;// = 32;
	cl_uint deviceComputeCapability;
	cl_uint maxComputeUnits;
	cl_ulong maxLocalMemory;// = -1;
	cl_uint vectorwidth;
	//---
	#ifdef DEBUGGTOPENCL
		float time_initialize_opencl;// = 0;
	#endif
	//---
	size_t kernels_size;// = 10 = Klength
	cl_kernel *kernels;
	size_t *kernelsGlobalSize;// = 10;
	size_t *kernelsLocalSize;// = 10;
	#ifdef DEBUGGTOPENCL
		oclDebugInfo *kernelsInfo; // {k0, k1,k2,k3,k5,k6p1n8,k6p1n4,k6p1n2,k6p2,k7}
	#endif
	//---
	size_t mems_size;// = 30 = Mlength;
	cl_mem *mems;
	#ifdef DEBUGGTOPENCL
		oclDebugInfo *memsH2DInfo; // {pOrg,piRef,ruiDist}
		oclDebugInfo *memsD2HInfo; // {k2Res,FinalRes}
		oclDebugInfo *memsD2DInfo; // {k0c0,k0c1,k0c2,k0c3,k0c4}
		oclDebugInfo *memsMapInfo; // {map}
		oclDebugInfo *memsUnmapInfo; // {unmap}
	#endif
} oclDeviceData;

///////////////////////////////
oclDeviceData *gtOpenClDeviceData = NULL;
size_t gtOpenClNumDevices = 0;
///////////////////////////////

////////////////////////////////////////
//  OpenCL Quick Getters and Setters  //
////////////////////////////////////////
size_t gtOpenclGetNumDevices(){ return gtOpenClNumDevices; }
cl_device_id     *gtOpenclGetDevice  (int device){ return &gtOpenClDeviceData[device].device; }
cl_context       *gtOpenclGetContext (int device){ return &gtOpenClDeviceData[device].context; }
cl_command_queue *gtOpenclGetQueue   (int device){ return &gtOpenClDeviceData[device].queue; }
cl_program       *gtOpenclGetProgram (int device){ return &gtOpenClDeviceData[device].program; }
cl_uint    gtOpenclGetWarpSize                (int device){ return gtOpenClDeviceData[device].warpSize; }
cl_uint    gtOpenclGetDeviceComputeCapability (int device){ return gtOpenClDeviceData[device].deviceComputeCapability; }
cl_uint    gtOpenclGetMaxComputeUnits         (int device){ return gtOpenClDeviceData[device].maxComputeUnits; }
cl_ulong   gtOpenclGetMaxLocalMemory          (int device){ return gtOpenClDeviceData[device].maxLocalMemory; }
cl_uint    gtOpenclGetVectorWidth             (int device){ return gtOpenClDeviceData[device].vectorwidth; }
//---
cl_kernel    *gtOpenclGetKernel    (int device, enumKernel index){ return &gtOpenClDeviceData[device].kernels[index]; }
size_t *gtOpenclGetKernelGlobalSize(int device, enumKernel index)             { return &gtOpenClDeviceData[device].kernelsGlobalSize[index]; }
void    gtOpenclSetKernelGlobalSize(int device, enumKernel index, size_t size){ gtOpenClDeviceData[device].kernelsGlobalSize[index] = size; }
size_t *gtOpenclGetKernelLocalSize (int device, enumKernel index)             { return &gtOpenClDeviceData[device].kernelsLocalSize[index]; }
void    gtOpenclSetKernelLocalSize (int device, enumKernel index, size_t size){ gtOpenClDeviceData[device].kernelsLocalSize[index] = size; }
//---
cl_mem *gtOpenclGetMem(int device, enumMem index){ return &gtOpenClDeviceData[device].mems[index]; }














void buildProgram(int deviceIndex){
	///////////////////////////////////////////////
	////            Program Options            ////
	///////////////////////////////////////////////
	//options for compile program
	char *options = (char*)malloc(1024*sizeof(char));

	sprintf(options, //-cl-no-signed-zeros 
		"-w -Werror -cl-std=CL%.1f -cl-denorms-are-zero -cl-finite-math-only -D %s -D DATATYPEV=%s",
		1.1, (USE_GPU==1)?"FOR_OPENCL_GPU":"FOR_OPENCL_CPU", (USE_FLOAT==1)?"float":"int");

	#if USE_GPU == 1
		sprintf(options, "%s -D ABS=%s", options, ((USE_FLOAT==1)?"fabs":"abs"));
	#else
		#if USE_FLOAT==1
			if(gtOpenclGetVectorWidth(deviceIndex)>1)
				sprintf(options, "%s%u -D MATRIXperBLOCK=%u -D ABS(x)=fabs(x)", options, gtOpenclGetVectorWidth(deviceIndex), gtOpenclGetVectorWidth(deviceIndex));
			else
				sprintf(options, "%s -D MATRIXperBLOCK=%u -D ABS(x)=fabs(x)", options, gtOpenclGetVectorWidth(deviceIndex));
		#else
			if(gtOpenclGetVectorWidth(deviceIndex)>1)
				sprintf(options, "%s%u -D MATRIXperBLOCK=%u -D ABS(x)=convert_int%d(abs(x))", options, gtOpenclGetVectorWidth(deviceIndex), gtOpenclGetVectorWidth(deviceIndex), gtOpenclGetVectorWidth(deviceIndex));
			else
				sprintf(options, "%s -D MATRIXperBLOCK=%u -D ABS(x)=abs(x)", options, gtOpenclGetVectorWidth(deviceIndex));
		#endif
	#endif
	
	for (int i = 0; i < Klength; ++i)
	{
		sprintf(options,
			"%s -D K%dG=%zu -D K%dL=%zu ",
			options,
			i,
			*gtOpenclGetKernelGlobalSize( deviceIndex, i),
			i,
			*gtOpenclGetKernelLocalSize(  deviceIndex, i)
		);
	}


	printf(". OpenCL Build Options = %s\n\n",options);

	////////////////////////////////////////////////
	////             build program              ////
	////////////////////////////////////////////////
	int err = clBuildProgram(*gtOpenclGetProgram(deviceIndex), 1, gtOpenclGetDevice(deviceIndex), options, NULL, NULL);
	free(options);
	if (err != CL_SUCCESS) {    
		size_t len = 0;
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(*gtOpenclGetProgram(deviceIndex), *gtOpenclGetDevice(deviceIndex), CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer = calloc(len, sizeof(char));
		clGetProgramBuildInfo(*gtOpenclGetProgram(deviceIndex), *gtOpenclGetDevice(deviceIndex), CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		printf("LOG>>>>>>>>>>>>>>>:\n%s\n<<<<<<<<<<<<<<<LOG\n", buffer);
		fflush(stdout);
		free(buffer);

		HANDLE_ERROR(clReleaseProgram(*gtOpenclGetProgram(deviceIndex)),"Error release Program");
		HANDLE_ERROR(clReleaseCommandQueue(*gtOpenclGetQueue(deviceIndex)),"Error release Command Queue");
		HANDLE_ERROR(clReleaseContext(*gtOpenclGetContext(deviceIndex)),"Error release Context");
		
		fflush(stdout);
		exit(1);
	}
	#ifdef DEBUGGTOPENCL
		size_t len = 0;
		clGetProgramBuildInfo(*gtOpenclGetProgram(deviceIndex), *gtOpenclGetDevice(deviceIndex), CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char *buffer = calloc(len, sizeof(char));
		clGetProgramBuildInfo(*gtOpenclGetProgram(deviceIndex), *gtOpenclGetDevice(deviceIndex), CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		printf("LOG>>>>>>>>>>>>>>>:%s\n<<<<<<<<<<<<<<<LOG\n", buffer);
		fflush(stdout);
		free(buffer);
	#endif
}



char* loadProgramSource(const char *filename){
	struct stat statbuf;
	FILE *file;
	size_t size_read;
	char* program_source = NULL;

	file = fopen(filename, "r");
	if (!file)
		return NULL;

	stat(filename, &statbuf);
	program_source = (char *) malloc(statbuf.st_size + 1);
	size_read = fread(program_source, statbuf.st_size, 1, file);
	
	if(size_read != 1 /*|| strlen(program_source) != statbuf.st_size*/){
		printf("Error reading program source! read=%lu expected=%ld\n",size_read,statbuf.st_size);
		free(program_source);
		return NULL;
	}
	program_source[statbuf.st_size] = '\0';
	fclose(file);
	
	return program_source;
}


void gtOpenclInitDeviceDataStructure(int numDevices){
	//---set default launch sizes
	size_t def[4] = {1,1,1,1};
	//--- init data structure
	gtOpenClNumDevices = numDevices;
	gtOpenClDeviceData = (oclDeviceData*) calloc(numDevices, sizeof(oclDeviceData));
	//--- init inner
	int i = 0;
	for (; i < numDevices; ++i){
		//gtOpenClDeviceData[i].device = NULL;
		//gtOpenClDeviceData[i].context = NULL;
		//gtOpenClDeviceData[i].queue = NULL;
		//gtOpenClDeviceData[i].program = NULL;
		//---
		gtOpenClDeviceData[i].warpSize = 32;
		gtOpenClDeviceData[i].deviceComputeCapability = -1;
		gtOpenClDeviceData[i].maxComputeUnits = -1;
		gtOpenClDeviceData[i].maxLocalMemory = -1;
		gtOpenClDeviceData[i].vectorwidth = 0;
		//---
		#ifdef DEBUGGTOPENCL
			gtOpenClDeviceData[i].time_initialize_opencl = 0;
		#endif
		//---
		gtOpenClDeviceData[i].kernels_size = Klength;//10
		gtOpenClDeviceData[i].kernels = (cl_kernel*) calloc(Klength, sizeof(cl_kernel));
		gtOpenClDeviceData[i].kernelsGlobalSize = (size_t*) calloc(Klength, sizeof(size_t));
		gtOpenClDeviceData[i].kernelsLocalSize = (size_t*) calloc(Klength, sizeof(size_t));
		int j = 0;
		for (; j < Klength; ++j){
			gtOpenClDeviceData[i].kernelsGlobalSize[j] = def[j*2];
			gtOpenClDeviceData[i].kernelsLocalSize[j] = def[j*2+1];
		}
		#ifdef DEBUGGTOPENCL
			gtOpenClDeviceData[i].kernelsInfo = (oclDebugInfo*) calloc(Klength, sizeof(oclDebugInfo));
		#endif
		//---
		gtOpenClDeviceData[i].mems_size = Mlength;
		gtOpenClDeviceData[i].mems = (cl_mem*) calloc(Mlength, sizeof(cl_mem));
		#ifdef DEBUGGTOPENCL
			gtOpenClDeviceData[i].memsH2DInfo = (oclDebugInfo*) calloc(Mlength, sizeof(oclDebugInfo));
			gtOpenClDeviceData[i].memsD2HInfo = (oclDebugInfo*) calloc(Mlength, sizeof(oclDebugInfo));
			gtOpenClDeviceData[i].memsD2DInfo = (oclDebugInfo*) calloc(Mlength, sizeof(oclDebugInfo));
			gtOpenClDeviceData[i].memsMapInfo = (oclDebugInfo*) calloc(Mlength, sizeof(oclDebugInfo));
			gtOpenClDeviceData[i].memsUnmapInfo = (oclDebugInfo*) calloc(Mlength, sizeof(oclDebugInfo));
		#endif
	}
}




void gtOpenclInitialize(){
	#ifdef DEBUGGTOPENCL
		struct timeval start, end;
		gettimeofday(&start, NULL);
	#endif

	int err;
	unsigned int i, j;

	////////////////////////////////////////////////
	////              select device             ////
	////////////////////////////////////////////////
	unsigned int numPlatforms = 0;
	unsigned int deviceCount = 0;
	cl_device_id *devices = NULL;
	HANDLE_ERROR( clGetPlatformIDs(0, NULL, &numPlatforms) , "Failure in clGetPlatformIDs!");
	cl_platform_id *platforms = (cl_platform_id*)malloc( numPlatforms * sizeof(cl_platform_id) );
	HANDLE_ERROR( clGetPlatformIDs(numPlatforms, platforms, NULL) , "Failure in clGetPlatformIDs!");

	for (i = 0; i < numPlatforms; ++i)  {
		clGetDeviceIDs(platforms[i], (USE_GPU==1) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &deviceCount);
		if(deviceCount > 0){
			deviceCount=1;
			gtOpenclInitDeviceDataStructure(deviceCount);
			devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
			clGetDeviceIDs(platforms[i], (USE_GPU==1) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, deviceCount, devices, NULL);
			for (j = 0; j < deviceCount; ++j){
				gtOpenClDeviceData[j].device = devices[j];
			}
			break;
		}
	}
	free(platforms);
	if(devices == NULL){
		HANDLE_ERROR(0, "Could not find a device");
	}
	printf(". Found %d device%c\n", deviceCount, (deviceCount==1)?' ':'s');

	int d = 0;
	for (; d < deviceCount; ++d){

				size_t valueSize;
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_NAME, 0, NULL, &valueSize);
				char *value = (char*) malloc(valueSize);
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_NAME, valueSize, value, NULL);
				printf(". OpenCL Device[%d]: %s\n", d, value);
				free(value);

				cl_device_type type;
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
				if( type & CL_DEVICE_TYPE_CPU )
					printf(".        DeviceType: CPU\n");
				if( type & CL_DEVICE_TYPE_GPU )
					printf(".        DeviceType: GPU\n");
				if( type & CL_DEVICE_TYPE_ACCELERATOR )
					printf(".        DeviceType: ACCELERATOR\n");
				if( type & CL_DEVICE_TYPE_DEFAULT )
					printf(".        DeviceType: DEFAULT\n");

				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_VENDOR, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_VENDOR, valueSize, value, NULL);
				printf(".        DeviceVendor: %s\n", value);
				free(value);

				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_VERSION, valueSize, value, NULL);
				printf(".        DeviceVersion: %s\n", value);
				free(value);

						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
						value = (char*) malloc(valueSize);
						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_EXTENSIONS, valueSize, value, NULL);
						printf(".        DeviceExtensions: %s\n", value);
						free(value);

						cl_uint ccmajor, ccminor;
						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &ccmajor, NULL);
						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &ccminor, NULL);
						gtOpenClDeviceData[d].deviceComputeCapability = (10*ccmajor)+ccminor;
						printf(".        DeviceComputeCapability: %u.%u  (%d)\n", ccmajor, ccminor, gtOpenClDeviceData[d].deviceComputeCapability );

						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_WARP_SIZE_NV, sizeof(cl_uint), &(gtOpenClDeviceData[d].warpSize), NULL);
						printf(".        DeviceWarpSize: %u\n", gtOpenClDeviceData[d].warpSize);

						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &(gtOpenClDeviceData[d].maxComputeUnits), NULL);
						printf(".        DeviceMaxComputeUnits: %u\n", gtOpenClDeviceData[d].maxComputeUnits);

						clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &(gtOpenClDeviceData[d].maxLocalMemory), NULL);
						printf(".        DeviceLocalMemory: %lu\n", gtOpenClDeviceData[d].maxLocalMemory);

						#if USE_FLOAT == 1
							clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &(gtOpenClDeviceData[d].vectorwidth), NULL);
							printf(".        VectorSupport: float%u\n", gtOpenClDeviceData[d].vectorwidth);
						#else
							clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(cl_uint), &(gtOpenClDeviceData[d].vectorwidth), NULL);
							printf(".        VectorSupport: int%u\n", gtOpenClDeviceData[d].vectorwidth);
						#endif




				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
				value = (char*) malloc(valueSize);
				clGetDeviceInfo(gtOpenClDeviceData[d].device, CL_DRIVER_VERSION, valueSize, value, NULL);
				printf(".        DriverVersion: %s\n", value);
				free(value);

				////////////////////////////////////////////////
				////             create context             ////
				////////////////////////////////////////////////
				gtOpenClDeviceData[d].context = clCreateContext(0, 1, &gtOpenClDeviceData[d].device, NULL, NULL, &err);
				HANDLE_ERROR(err, "Failed to create a compute context!");
				
				////////////////////////////////////////////////
				////          create command queue          ////
				////////////////////////////////////////////////
				#ifdef DEBUGGTOPENCL
					gtOpenClDeviceData[d].queue = clCreateCommandQueue(gtOpenClDeviceData[d].context, gtOpenClDeviceData[d].device, CL_QUEUE_PROFILING_ENABLE, &err);
				#else
					gtOpenClDeviceData[d].queue = clCreateCommandQueue(gtOpenClDeviceData[d].context, gtOpenClDeviceData[d].device, 0 /*IN-ORDER*/, &err);
				#endif
				
				HANDLE_ERROR(err, "Failed to create a command queue!");
				
				////////////////////////////////////////////////
				////             create program             ////
				////////////////////////////////////////////////
				char *kernels_source = loadProgramSource("OpenCLKernels.cl");
				if(!kernels_source) {
					printf("Error: Failed to load compute program from file!\n");
					fflush(stdout);
					exit( EXIT_FAILURE );
				}

				gtOpenClDeviceData[d].program = clCreateProgramWithSource(gtOpenClDeviceData[d].context, 1, (const char **) & kernels_source, NULL, &err);
				HANDLE_ERROR(err,"Failed to create compute program!");
				buildProgram(d);
				free(kernels_source);

				////////////////////////////////////////////////
				////           instanceat kernels           ////
				////////////////////////////////////////////////
				for (i = 0; i < Klength; ++i){
					gtOpenClDeviceData[d].kernels[i] = clCreateKernel(gtOpenClDeviceData[d].program, enumKernelName[i], &err);
					HANDLE_ERROR(err,"Failed to create compute %s!",enumKernelDisplayName[i]);
				}

	} //for devices

	////////////////////////////////////////////////
	////              free/release              ////
	////////////////////////////////////////////////
	free(devices);
	
	#ifdef DEBUGGTOPENCL
		gettimeofday(&end, NULL);
		gtOpenClDeviceData[0].time_initialize_opencl = ((end.tv_sec  - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000.0);
	#endif
}





void gtOpenclRelease() {
	unsigned int i,j,d;
	#ifdef DEBUGGTOPENCL
		////////////////////////////////////////////////
		////            output statistics           ////
		////////////////////////////////////////////////
		float total = gtOpenClDeviceData[0].time_initialize_opencl/1000;
		printf("\nInitialize OpenCL: %.6f s\n",total);
		
		for (d = 0; d < gtOpenClNumDevices; ++d){
					float global = 0, local, sub;
					int nr_sub;
					float Stime_queued_submited=0, Stime_submited_running=0, Stime_running=0;
					int Snr_queued_submited=0, Snr_submited_running=0, Snr_running=0;
					printf("\nTimes:             ----------[ device %d ]----------\n",d);

					for (i = 0; i < Klength; ++i){
						global += gtOpenClDeviceData[d].kernelsInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].kernelsInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].kernelsInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].kernelsInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].kernelsInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].kernelsInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].kernelsInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].kernelsInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].kernelsInfo[i].nr_running;
					}

					for (i = 0; i < Mlength; ++i){
						global += gtOpenClDeviceData[d].memsH2DInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].memsH2DInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].memsH2DInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].memsH2DInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].memsH2DInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].memsH2DInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].memsH2DInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].memsH2DInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].memsH2DInfo[i].nr_running;
						
						global += gtOpenClDeviceData[d].memsD2HInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].memsD2HInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].memsD2HInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].memsD2HInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].memsD2HInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].memsD2HInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].memsD2HInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].memsD2HInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].memsD2HInfo[i].nr_running;
						
						global += gtOpenClDeviceData[d].memsD2DInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].memsD2DInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].memsD2DInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].memsD2DInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].memsD2DInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].memsD2DInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].memsD2DInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].memsD2DInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].memsD2DInfo[i].nr_running;

						global += gtOpenClDeviceData[d].memsMapInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].memsMapInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].memsMapInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].memsMapInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].memsMapInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].memsMapInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].memsMapInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].memsMapInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].memsMapInfo[i].nr_running;

						global += gtOpenClDeviceData[d].memsUnmapInfo[i].time_queued_submited;
						global += gtOpenClDeviceData[d].memsUnmapInfo[i].time_submited_running;
						global += gtOpenClDeviceData[d].memsUnmapInfo[i].time_running;
							Stime_queued_submited += gtOpenClDeviceData[d].memsUnmapInfo[i].time_queued_submited;
							Stime_submited_running += gtOpenClDeviceData[d].memsUnmapInfo[i].time_submited_running;
							Stime_running += gtOpenClDeviceData[d].memsUnmapInfo[i].time_running;
							Snr_queued_submited += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_queued_submited;
							Snr_submited_running += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_submited_running;
							Snr_running += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_running;
					}

					global = global/1000;
					printf("  total:                  %.6f\n", global);

					total = Stime_queued_submited/1000;
					printf("    Queued to submited:     %.6f s (%.2f%% - #:%d)\n",total,total/global*100,Snr_queued_submited);
						local = 0;
						for (i = 0; i < Klength; ++i) local += gtOpenClDeviceData[d].kernelsInfo[i].time_queued_submited;
						local /= 1000;
						printf("      q.Kernel:               %.6f s (%.2f%%)\n", local, local/total*100);
						for (i = 0; i < Klength; ++i){
							sub = gtOpenClDeviceData[d].kernelsInfo[i].time_queued_submited/1000;
							printf("        q.%-22s%.6f s (%.2f%% - #:%d)\n", enumKernelDisplayName[i], sub, sub/local*100, gtOpenClDeviceData[d].kernelsInfo[i].nr_queued_submited);
						}
						local = 0;
						for (i = 0; i < Mlength; ++i)
							local += (gtOpenClDeviceData[d].memsH2DInfo[i].time_queued_submited 
								    + gtOpenClDeviceData[d].memsD2HInfo[i].time_queued_submited 
								    + gtOpenClDeviceData[d].memsD2DInfo[i].time_queued_submited 
								    + gtOpenClDeviceData[d].memsMapInfo[i].time_queued_submited 
								    + gtOpenClDeviceData[d].memsUnmapInfo[i].time_queued_submited)/1000;
						printf("      q.Memory:               %.6f s (%.2f%%)\n", local, local/total*100);
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsH2DInfo[i].time_queued_submited/1000;
								nr_sub += gtOpenClDeviceData[d].memsH2DInfo[i].nr_queued_submited;
							} printf("        q.H2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsH2DInfo[i].time_queued_submited/1000; nr_sub = gtOpenClDeviceData[d].memsH2DInfo[i].nr_queued_submited;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2HInfo[i].time_queued_submited/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2HInfo[i].nr_queued_submited;
							} printf("        q.D2H:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2HInfo[i].time_queued_submited/1000; nr_sub = gtOpenClDeviceData[d].memsD2HInfo[i].nr_queued_submited;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2DInfo[i].time_queued_submited/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2DInfo[i].nr_queued_submited;
							} printf("        q.D2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2DInfo[i].time_queued_submited/1000; nr_sub = gtOpenClDeviceData[d].memsD2DInfo[i].nr_queued_submited;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsMapInfo[i].time_queued_submited/1000;
								nr_sub += gtOpenClDeviceData[d].memsMapInfo[i].nr_queued_submited;
							} printf("        q.Map:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsMapInfo[i].time_queued_submited/1000; nr_sub = gtOpenClDeviceData[d].memsMapInfo[i].nr_queued_submited;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsUnmapInfo[i].time_queued_submited/1000;
								nr_sub += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_queued_submited;
							} printf("        q.Unmap:                %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsUnmapInfo[i].time_queued_submited/1000; nr_sub = gtOpenClDeviceData[d].memsUnmapInfo[i].nr_queued_submited;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}

					total = Stime_submited_running/1000;
					printf("    Submited to running:    %.6f s (%.2f%% - #:%d)\n",total,total/global*100,Snr_submited_running);
						local = 0;
						for (i = 0; i < Klength; ++i) local += gtOpenClDeviceData[d].kernelsInfo[i].time_submited_running;
						local /= 1000;
						printf("      s.Kernel:               %.6f s (%.2f%%)\n", local, local/total*100);
						for (i = 0; i < Klength; ++i){
							sub = gtOpenClDeviceData[d].kernelsInfo[i].time_submited_running/1000;
							printf("        s.%-22s%.6f s (%.2f%% - #:%d)\n", enumKernelDisplayName[i], sub, sub/local*100, gtOpenClDeviceData[d].kernelsInfo[i].nr_submited_running);
						}
						local = 0;
						for (i = 0; i < Mlength; ++i)
							local += (gtOpenClDeviceData[d].memsH2DInfo[i].time_submited_running 
								    + gtOpenClDeviceData[d].memsD2HInfo[i].time_submited_running 
								    + gtOpenClDeviceData[d].memsD2DInfo[i].time_submited_running 
								    + gtOpenClDeviceData[d].memsMapInfo[i].time_submited_running 
								    + gtOpenClDeviceData[d].memsUnmapInfo[i].time_submited_running)/1000;
						printf("      s.Memory:               %.6f s (%.2f%%)\n", local, local/total*100);
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsH2DInfo[i].time_submited_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsH2DInfo[i].nr_submited_running;
							} printf("        s.H2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsH2DInfo[i].time_submited_running/1000; nr_sub = gtOpenClDeviceData[d].memsH2DInfo[i].nr_submited_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2HInfo[i].time_submited_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2HInfo[i].nr_submited_running;
							} printf("        s.D2H:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2HInfo[i].time_submited_running/1000; nr_sub = gtOpenClDeviceData[d].memsD2HInfo[i].nr_submited_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2DInfo[i].time_submited_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2DInfo[i].nr_submited_running;
							} printf("        s.D2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2DInfo[i].time_submited_running/1000; nr_sub = gtOpenClDeviceData[d].memsD2DInfo[i].nr_submited_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsMapInfo[i].time_submited_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsMapInfo[i].nr_submited_running;
							} printf("        s.Map:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsMapInfo[i].time_submited_running/1000; nr_sub = gtOpenClDeviceData[d].memsMapInfo[i].nr_submited_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsUnmapInfo[i].time_submited_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_submited_running;
							} printf("        s.Unmap:                %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsUnmapInfo[i].time_submited_running/1000; nr_sub = gtOpenClDeviceData[d].memsUnmapInfo[i].nr_submited_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}

					total = Stime_running/1000;
					printf("    Running to end:         %.6f s (%.2f%% - #:%d)\n", total,total/global*100,Snr_running);
						local = 0;
						for (i = 0; i < Klength; ++i) local += gtOpenClDeviceData[d].kernelsInfo[i].time_running;
						local /= 1000;
						printf("      r.Kernel:               %.6f s (%.2f%%)\n", local, local/total*100);
						for (i = 0; i < Klength; ++i){
							sub = gtOpenClDeviceData[d].kernelsInfo[i].time_running/1000;
							printf("        r.%-22s%.6f s (%.2f%% - #:%d)\n", enumKernelDisplayName[i], sub, sub/local*100, gtOpenClDeviceData[d].kernelsInfo[i].nr_running);
						}
						local = 0;
						for (i = 0; i < Mlength; ++i)
							local += (gtOpenClDeviceData[d].memsH2DInfo[i].time_running 
								    + gtOpenClDeviceData[d].memsD2HInfo[i].time_running 
								    + gtOpenClDeviceData[d].memsD2DInfo[i].time_running 
								    + gtOpenClDeviceData[d].memsMapInfo[i].time_running 
								    + gtOpenClDeviceData[d].memsUnmapInfo[i].time_running)/1000;
						printf("      r.Memory:               %.6f s (%.2f%%)\n", local, local/total*100);
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsH2DInfo[i].time_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsH2DInfo[i].nr_running;
							} printf("        s.H2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsH2DInfo[i].time_running/1000; nr_sub = gtOpenClDeviceData[d].memsH2DInfo[i].nr_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2HInfo[i].time_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2HInfo[i].nr_running;
							} printf("        s.D2H:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2HInfo[i].time_running/1000; nr_sub = gtOpenClDeviceData[d].memsD2HInfo[i].nr_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsD2DInfo[i].time_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsD2DInfo[i].nr_running;
							} printf("        s.D2D:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsD2DInfo[i].time_running/1000; nr_sub = gtOpenClDeviceData[d].memsD2DInfo[i].nr_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsMapInfo[i].time_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsMapInfo[i].nr_running;
							} printf("        s.Map:                  %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsMapInfo[i].time_running/1000; nr_sub = gtOpenClDeviceData[d].memsMapInfo[i].nr_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
							for (sub = 0, nr_sub = 0, i = 0; i < Mlength; ++i) {
								sub += gtOpenClDeviceData[d].memsUnmapInfo[i].time_running/1000;
								nr_sub += gtOpenClDeviceData[d].memsUnmapInfo[i].nr_running;
							} printf("        s.Unmap:                %.6f s (%.2f%% - #:%d)\n", sub, sub/local*100, nr_sub);
								for (i = 0; i < Mlength; ++i) {
									sub = gtOpenClDeviceData[d].memsUnmapInfo[i].time_running/1000; nr_sub = gtOpenClDeviceData[d].memsUnmapInfo[i].nr_running;
									if(nr_sub != 0) printf("          q.%-22s%.6f s (%.2f%% - #:%d)\n", enumMemDisplayName[i], sub, sub/local*100, nr_sub);
								}
		}//for devices
	#endif

	////////////////////////////////////////////////
	////              free/release              ////
	////////////////////////////////////////////////
	for (i = 0; i < gtOpenClNumDevices; ++i){
		for (j = 0; j < Klength; ++j){
			HANDLE_ERROR(clReleaseKernel(gtOpenClDeviceData[i].kernels[j]),"Error release kernel %s", enumKernelDisplayName[j]);
		}
		HANDLE_ERROR(clReleaseProgram(gtOpenClDeviceData[i].program),"Error release Program");
		HANDLE_ERROR(clReleaseCommandQueue(gtOpenClDeviceData[i].queue),"Error release Command Queue");
		HANDLE_ERROR(clReleaseContext(gtOpenClDeviceData[i].context),"Error release Context");
		for (j = 0; j < Mlength; ++j){
			HANDLE_ERROR(clReleaseMemObject(gtOpenClDeviceData[i].mems[j]),"Error release MemObject %s", enumMemDisplayName[j]);
		}
	}
	for (i = 0; i < gtOpenClNumDevices; ++i){
		free(gtOpenClDeviceData[i].kernels);
		free(gtOpenClDeviceData[i].kernelsGlobalSize);
		free(gtOpenClDeviceData[i].kernelsLocalSize);
		free(gtOpenClDeviceData[i].mems);
		#ifdef DEBUGGTOPENCL
			free(gtOpenClDeviceData[i].kernelsInfo);
			free(gtOpenClDeviceData[i].memsH2DInfo);
			free(gtOpenClDeviceData[i].memsD2HInfo);
			free(gtOpenClDeviceData[i].memsD2DInfo);
			free(gtOpenClDeviceData[i].memsMapInfo);
			free(gtOpenClDeviceData[i].memsUnmapInfo);
		#endif
	}
	free(gtOpenClDeviceData);

}





void gtOpenclHandleError(int err, const char *file, int line, const char *fmt, ... ){
	if (err != CL_SUCCESS) {
		fprintf(stderr, "\n[OPENCL_ERROR");

		if(err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
			fprintf(stderr, ":CL_MEM_OBJECT_ALLOCATION_FAILURE");
		
		if(err == CL_INVALID_COMMAND_QUEUE)
			fprintf(stderr, ":CL_INVALID_COMMAND_QUEUE");
		
		if(err == CL_INVALID_CONTEXT)
			fprintf(stderr, ":CL_INVALID_CONTEXT");
		
		if(err == CL_INVALID_EVENT_WAIT_LIST)
			fprintf(stderr, ":CL_INVALID_EVENT_WAIT_LIST");
		
		if(err == CL_INVALID_MEM_OBJECT)
			fprintf(stderr, ":CL_INVALID_MEM_OBJECT");
		
		if(err == CL_INVALID_VALUE)
			fprintf(stderr, ":CL_INVALID_VALUE");
		
		if(err == CL_MEM_OBJECT_ALLOCATION_FAILURE)
			fprintf(stderr, ":CL_MEM_OBJECT_ALLOCATION_FAILURE");
		
		if(err == CL_OUT_OF_RESOURCES)
			fprintf(stderr, ":CL_OUT_OF_RESOURCES");

		//----
		va_list ap;
		va_start(ap, fmt);
		//----

    	fprintf(stderr, "]:\n\t'");
    	vfprintf(stderr, fmt, ap);
    	fprintf(stderr, "'(%d)\n\t in '%s' at line '%d'\n", err, file, line);
		va_end(ap);
    	fflush(stderr);
		exit( EXIT_FAILURE );
	}
}





#ifdef DEBUGGTOPENCL

	//calc time between kernels states
	//   Queued to Submit
	//   Submit to Start
	void gtOpenclDebugEventPreExecutionTimeK(cl_event event, int deviceIndex, enumKernel kernel){
		return;
		cl_ulong queued, submit, start, end;
		float millisecondsQS, millisecondsSR;
		
		HANDLE_ERROR(clWaitForEvents(1, &event), "ERROR clWaitForEvents");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong), &queued, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong), &submit, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL),"ERROR clGetEventProfilingInfo");
		
		if(submit < queued) {
			printf("BUG[the laws of the universe have change]:%d: submit=%lu queued=%lu\n", __LINE__,submit,queued);
			fflush(stdout);
			exit(0);
		}
		
		if(start < submit) {
			printf("BUG[the laws of the universe have change]:%d: start=%lu submit=%lu\n", __LINE__,start,submit);
			fflush(stdout);
			exit(0);
		}
		
		millisecondsQS = (submit - queued) * 0.000001;
		millisecondsSR = (start - submit) * 0.000001;
		
		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].time_queued_submited += millisecondsQS;
		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].nr_queued_submited ++;
		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].time_submited_running += millisecondsSR;
		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].nr_submited_running ++;
	}

	//calc time between kernels states
	//   Queued to Submit
	//   Submit to Start
	void gtOpenclDebugEventPreExecutionTimeM(cl_event event, int deviceIndex, enumMem mem, enumMemTransferType type){
		return;
		cl_ulong queued, submit, start, end;
		float millisecondsQS, millisecondsSR;
		
		HANDLE_ERROR(clWaitForEvents(1, &event), "ERROR clWaitForEvents");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,sizeof(cl_ulong), &queued, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,sizeof(cl_ulong), &submit, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL),"ERROR clGetEventProfilingInfo");
		
		if(submit < queued) {
			printf("BUG[the laws of the universe have change]:%d: submit=%lu queued=%lu\n", __LINE__,submit,queued);
			fflush(stdout);
			exit(0);
		}
		
		if(start < submit) {
			printf("BUG[the laws of the universe have change]:%d: start=%lu submit=%lu\n", __LINE__,start,submit);
			fflush(stdout);
			exit(0);
		}
		
		millisecondsQS = (submit - queued) * 0.000001;
		millisecondsSR = (start - submit) * 0.000001;
		
		if(type == cpy_h2d){
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].time_queued_submited += millisecondsQS;
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].nr_queued_submited ++;
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].time_submited_running += millisecondsSR;
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].nr_submited_running ++;
		}else if(type == cpy_d2h){
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].time_queued_submited += millisecondsQS;
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].nr_queued_submited ++;
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].time_submited_running += millisecondsSR;
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].nr_submited_running ++;
		}else if(type == cpy_d2d){
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].time_queued_submited += millisecondsQS;
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].nr_queued_submited ++;
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].time_submited_running += millisecondsSR;
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].nr_submited_running ++;
		}else if(type == maps){
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].time_queued_submited += millisecondsQS;
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].nr_queued_submited ++;
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].time_submited_running += millisecondsSR;
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].nr_submited_running ++;
		}else if(type == unmaps){
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].time_queued_submited += millisecondsQS;
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].nr_queued_submited ++;
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].time_submited_running += millisecondsSR;
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].nr_submited_running ++;
		}
	}



	//WARNING: this function release event at the end
	void gtOpenclDebugEventExecutionTimeK(cl_event event, int deviceIndex, enumKernel kernel){
		cl_ulong start, end;
		float milliseconds;
		
		HANDLE_ERROR(clWaitForEvents(1, &event), "ERROR clWaitForEvents");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL),"ERROR clGetEventProfilingInfo");
		milliseconds = (end - start) * 0.000001;
		
		if(milliseconds < 0) {
			printf("BUG[time travel is possible]:%d: execTime=%.7f\n", __LINE__, milliseconds);
			fflush(stdout);
			exit(0);
		}

		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].time_running += milliseconds;
		gtOpenClDeviceData[deviceIndex].kernelsInfo[kernel].nr_running ++;

		//release event
		HANDLE_ERROR(clReleaseEvent(event),"Error release event");
	}

	//WARNING: this function release event at the end
	void gtOpenclDebugEventExecutionTimeM(cl_event event, int deviceIndex, enumMem mem, enumMemTransferType type){
		cl_ulong start, end;
		float milliseconds;
		
		HANDLE_ERROR(clWaitForEvents(1, &event), "ERROR clWaitForEvents");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL),"ERROR clGetEventProfilingInfo");
		HANDLE_ERROR(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(cl_ulong), &start, NULL),"ERROR clGetEventProfilingInfo");
		milliseconds = (end - start) * 0.000001;
		
		if(milliseconds < 0) {
			printf("BUG[time travel is possible]:%d: execTime=%.7f\n", __LINE__, milliseconds);
			fflush(stdout);
			exit(0);
		}

		if(type == cpy_h2d){
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].time_running += milliseconds;
			gtOpenClDeviceData[deviceIndex].memsH2DInfo[mem].nr_running ++;
		}else if(type == cpy_d2h){
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].time_running += milliseconds;
			gtOpenClDeviceData[deviceIndex].memsD2HInfo[mem].nr_running ++;
		}else if(type == cpy_d2d){
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].time_running += milliseconds;
			gtOpenClDeviceData[deviceIndex].memsD2DInfo[mem].nr_running ++;
		}else if(type == maps){
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].time_running += milliseconds;
			gtOpenClDeviceData[deviceIndex].memsMapInfo[mem].nr_running ++;
		}else if(type == unmaps){
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].time_running += milliseconds;
			gtOpenClDeviceData[deviceIndex].memsUnmapInfo[mem].nr_running ++;
		}

		//release event
		HANDLE_ERROR(clReleaseEvent(event),"Error release event");
	}

#endif




// =====================================================================================
// Memory/Buffers Management
// =====================================================================================

void gtOpenclInitMem(enumMem memIndex, size_t sizeOfType, unsigned int elemLength){
	int err;
	int i = 0;
	for (; i < gtOpenClNumDevices; ++i){
		gtOpenClDeviceData[i].mems[memIndex] = clCreateBuffer(gtOpenClDeviceData[i].context, enumMemFlags[memIndex], 
			elemLength * sizeOfType, (void *)enumMemPointers[memIndex], &err);
		HANDLE_ERROR(err, "Couldn't create buffer %s", enumMemName[memIndex]);
	}
}







// =====================================================================================
// Default Kernel Launchs           (!) inline, thats why code in .h and not in .c (!)
// =====================================================================================

void gtOpenclLaunchKernel0(){
    int d=0;
    #ifdef DEBUGGTOPENCL
        cl_event _event;
    #endif
    //////////////////////////////////////
    HANDLE_ERROR(clSetKernelArg(*gtOpenclGetKernel(d,k0Prepare), 0,  sizeof(cl_mem),    gtOpenclGetMem(d,MemPrepare)  ),"Unable to set kernel argument");
    HANDLE_ERROR(clSetKernelArg(*gtOpenclGetKernel(d,k0Prepare), 1,  sizeof(cl_mem),    gtOpenclGetMem(d,MemInput)    ),"Unable to set kernel argument");
    //////////////////////////////////////
    #ifdef DEBUGGTOPENCL
        HANDLE_ERROR(clEnqueueNDRangeKernel(*gtOpenclGetQueue(d), *gtOpenclGetKernel(d,k0Prepare), 1, NULL, gtOpenclGetKernelGlobalSize(d, k0Prepare), gtOpenclGetKernelLocalSize(d, k0Prepare), 0, NULL, &_event), "Couldn't enqueue the kernel");
    #else
        HANDLE_ERROR(clEnqueueNDRangeKernel(*gtOpenclGetQueue(d), *gtOpenclGetKernel(d,k0Prepare), 1, NULL, gtOpenclGetKernelGlobalSize(d, k0Prepare), gtOpenclGetKernelLocalSize(d, k0Prepare), 0, NULL, NULL), "Couldn't enqueue the kernel"); 
    #endif
    #ifdef DEBUGGTOPENCL
        gtOpenclDebugEventPreExecutionTimeK(_event, d, k0Prepare);
        gtOpenclDebugEventExecutionTimeK(   _event, d, k0Prepare);
    #endif
    clFlush(*gtOpenclGetQueue(d));
    HANDLE_ERROR(clFinish(*gtOpenclGetQueue(d)),"");
}


void gtOpenclLaunchKernel1(){
    int d=0;
    #ifdef DEBUGGTOPENCL
        cl_event _event;
    #endif
    //////////////////////////////////////
    HANDLE_ERROR(clSetKernelArg(*gtOpenclGetKernel(d,k1Had), 0,  sizeof(cl_mem),    gtOpenclGetMem(d,MemInput)  ),"Unable to set kernel argument");
    HANDLE_ERROR(clSetKernelArg(*gtOpenclGetKernel(d,k1Had), 1,  sizeof(cl_mem),    gtOpenclGetMem(d,MemInput)  ),"Unable to set kernel argument");
    //////////////////////////////////////
    #ifdef DEBUGGTOPENCL
        HANDLE_ERROR(clEnqueueNDRangeKernel(*gtOpenclGetQueue(d), *gtOpenclGetKernel(d,k1Had), 1, NULL, gtOpenclGetKernelGlobalSize(d, k1Had), gtOpenclGetKernelLocalSize(d, k1Had), 0, NULL, &_event), "Couldn't enqueue the kernel");
    #else
        HANDLE_ERROR(clEnqueueNDRangeKernel(*gtOpenclGetQueue(d), *gtOpenclGetKernel(d,k1Had), 1, NULL, gtOpenclGetKernelGlobalSize(d, k1Had), gtOpenclGetKernelLocalSize(d, k1Had), 0, NULL, NULL), "Couldn't enqueue the kernel"); 
    #endif
    #ifdef DEBUGGTOPENCL
        gtOpenclDebugEventPreExecutionTimeK(_event, d, k1Had);
        gtOpenclDebugEventExecutionTimeK(   _event, d, k1Had);
    #endif
    clFlush(*gtOpenclGetQueue(d));
    HANDLE_ERROR(clFinish(*gtOpenclGetQueue(d)),"");
}

void gtOpenclCopyD2H(enumMem memIndex, void *buffer, size_t sizeofDataType, unsigned int elems_start, unsigned int elems_end){
    int d=0;
    #ifdef DEBUGGTOPENCL
        cl_event _event;
        HANDLE_ERROR(clEnqueueReadBuffer(*gtOpenclGetQueue(d), *gtOpenclGetMem(d,memIndex), CL_FALSE, sizeofDataType*elems_start, sizeofDataType*(elems_end-elems_start), buffer, 0, NULL, &_event), "Couldn't read the buffer");
    #else
        HANDLE_ERROR(clEnqueueReadBuffer(*gtOpenclGetQueue(d), *gtOpenclGetMem(d,memIndex), CL_TRUE,  sizeofDataType*elems_start, sizeofDataType*(elems_end-elems_start), buffer, 0, NULL, NULL), "Couldn't read the buffer");
    #endif

    #ifdef DEBUGGTOPENCL
        gtOpenclDebugEventPreExecutionTimeM(_event, d, memIndex, cpy_d2h);
        gtOpenclDebugEventExecutionTimeM(   _event, d, memIndex, cpy_d2h);
    #endif
    clFlush(*gtOpenclGetQueue(d));
    HANDLE_ERROR(clFinish(*gtOpenclGetQueue(d)),"");
}


void gtOpenclCopyH2D(enumMem memIndex, void *buffer, size_t sizeofDataType, unsigned int elems_start, unsigned int elems_end){
    int d=0;
    #ifdef DEBUGGTOPENCL
        cl_event _event;
        HANDLE_ERROR(clEnqueueWriteBuffer(*gtOpenclGetQueue(d), *gtOpenclGetMem(d,memIndex), CL_FALSE, sizeofDataType*elems_start, sizeofDataType*(elems_end-elems_start), buffer,  0, NULL, &_event), "Couldn't write to buffer");
    #else
        HANDLE_ERROR(clEnqueueWriteBuffer(*gtOpenclGetQueue(d), *gtOpenclGetMem(d,memIndex), CL_TRUE,  sizeofDataType*elems_start, sizeofDataType*(elems_end-elems_start), buffer,  0, NULL, NULL), "Couldn't write to buffer");
    #endif

    #ifdef DEBUGGTOPENCL
        gtOpenclDebugEventPreExecutionTimeM(_event, d, memIndex, cpy_h2d);
        gtOpenclDebugEventExecutionTimeM(   _event, d, memIndex, cpy_h2d);
    #endif
    clFlush(*gtOpenclGetQueue(d));
    HANDLE_ERROR(clFinish(*gtOpenclGetQueue(d)),"");
}



