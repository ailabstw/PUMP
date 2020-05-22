#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <string.h>

#include "fromCudnn.h"
#include "fromCublas.h"
#include "fromCuda.h"
#include "prefetch.h"
#include "global.h"
#include "hook.h"
#ifdef ANALYSIS
#include "analysis.h"
#endif

void *libcudnnHandle;
void *libcublasHandle;
void *libcudaHandle;
void *libcudartHandle;
void *libdlHandle;

bool inCudnnCublasInvocation = false;
bool inCudnnInvocation = false;
bool inCublasInvocation = false;

extern "C" {
  void *__libc_dlsym(void *map, const char *name);
  void *__libc_dlopen_mode(const char *name, int mode);
}

#define STRINGIFY(x) #x

__attribute__((constructor))
void hookInit(void) {
  libcudnnHandle = __libc_dlopen_mode("libcudnn.so", RTLD_LAZY);
  libcublasHandle = __libc_dlopen_mode("libcublas.so", RTLD_LAZY);
  libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
  libcudartHandle = __libc_dlopen_mode("libcudart.so", RTLD_LAZY);
  libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);

  _cudaEventCreateWithFlags = (cudaError_t(*)(cudaEvent_t*, unsigned int)) actualDlsym(libcudartHandle, STRINGIFY(cudaEventCreateWithFlags));
  _cudaEventDestroy = (cudaError_t(*)(cudaEvent_t)) actualDlsym(libcudartHandle, STRINGIFY(cudaEventDestroy));
  _cudaEventRecord = (cudaError_t(*)(cudaEvent_t, cudaStream_t)) actualDlsym(libcudartHandle, STRINGIFY(cudaEventRecord));
  _cudaGetDeviceProperties = (cudaError_t(*)(cudaDeviceProp*, int)) actualDlsym(libcudartHandle, STRINGIFY(cudaGetDeviceProperties));
  _cudaLaunchHostFunc = (cudaError_t(*)(cudaStream_t, cudaHostFn_t, void*)) actualDlsym(libcudartHandle, STRINGIFY(cudaLaunchHostFunc));
  _cudaMemPrefetchAsync = (cudaError_t(*)(const void*, size_t, int, cudaStream_t)) actualDlsym(libcudartHandle, STRINGIFY(cudaMemPrefetchAsync));
  _cudaStreamCreateWithPriority = (cudaError_t(*)(cudaStream_t*, unsigned int, int)) actualDlsym(libcudartHandle, STRINGIFY(cudaStreamCreateWithPriority));
  _cudaStreamWaitEvent = (cudaError_t(*)(cudaStream_t, cudaEvent_t, unsigned int)) actualDlsym(libcudartHandle, STRINGIFY(cudaStreamWaitEvent));
}

void *actualDlsym(void *handle, const char *symbol) {
  typedef decltype(&dlsym) funcType;
  funcType func = (funcType) __libc_dlsym(libdlHandle, "dlsym");

  void *ret = (*func)(handle, symbol);
  if (!ret) {
    ERROR("Error: Cannot load %s\n", symbol);
  }
  return ret;
}

void *dlsym(void *handle, const char *symbol) {
  if (strcmp(symbol, STRINGIFY(cuLaunchKernel)) == 0) {
    DEBUG("dlsym() hook %s\n", STRINGIFY(cuLaunchKernel));
    return (void*)(&cuLaunchKernel);
  } else if (strcmp(symbol, STRINGIFY(cuMemAlloc)) == 0) {
    DEBUG("dlsym() hook %s\n", STRINGIFY(cuMemAlloc));
    return (void*)(&cuMemAlloc);
  } else if (strcmp(symbol, STRINGIFY(cuMemAlloc_v2)) == 0) {
    DEBUG("dlsym() hook %s\n", STRINGIFY(cuMemAlloc_v2));
    return (void*)(&cuMemAlloc_v2);
  } else if (strcmp(symbol, STRINGIFY(cuMemAllocManaged)) == 0) {
    DEBUG("dlsym() hook %s\n", STRINGIFY(cuMemAllocManaged));
    return (void*)(&cuMemAllocManaged);
  } else if (strcmp(symbol, STRINGIFY(cuMemFree)) == 0) {
    DEBUG("dlsym() hook %s\n", STRINGIFY(cuMemFree));
    return (void*)(&cuMemFree);
  }

  DEBUG("dlsym() pass %s\n", symbol);
  return actualDlsym(handle, symbol);
}

#include "hook_api/cuda.cpp"
#include "hook_api/cudnn.cpp"
#include "hook_api/cublas.cpp"
