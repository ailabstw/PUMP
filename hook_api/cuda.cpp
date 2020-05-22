/* CUDA Runtime API part */

cudaError_t CUDAAPI cudaMallocManaged(void** devPtr, 
                                      size_t size, 
                                      unsigned int flags) {
  if (size == 0) {
    return cudaSuccess;
  }

  DEBUG("Enter cudaMallocManaged(), devPtr = %llx, size = %llu\n", (unsigned long long int) *devPtr, (unsigned long long int) size);

  cudaError_t (*func)(void**, size_t, unsigned int);
  func = (cudaError_t(*)(void**, size_t, unsigned int)) actualDlsym(libcudartHandle, STRINGIFY(cudaMallocManaged));

  cudaError_t ret = func(devPtr,
                         size,
                         flags);

  if (ret != cudaSuccess) {
    WARN("cudaMallocManaged() returns %d\n", (int) ret);
  } else {
    DEBUG("Exit cudaMallocManaged(), devPtr = %llx, size = %llu\n", (unsigned long long int) *devPtr, (unsigned long long int) size);
    fromCudaMalloc(*devPtr, size);
  }

  return ret;
}

cudaError_t CUDAAPI cudaMalloc(void** devPtr, 
                               size_t size) {
  return cudaMallocManaged(devPtr, size, cudaMemAttachGlobal);
}

cudaError_t CUDAAPI cudaFree(void* devPtr) {
  if (devPtr == 0) {
    return cudaSuccess;
  }

  DEBUG("Enter cudaFree(), devPtr = %llx\n", (unsigned long long int) devPtr);

  cudaError_t (*func)(void*);
  func = (cudaError_t(*)(void*)) actualDlsym(libcudartHandle, STRINGIFY(cudaFree));

  cudaError_t ret = func(devPtr);

  if (ret != cudaSuccess) {
    WARN("cudaFree() returns %d\n", (int) ret);
  } else {
    DEBUG("Exit cudaFree(), devPtr = %llx\n", (unsigned long long int) devPtr);
    fromCudaFree(devPtr);
  }

  return ret;
}

/* CUDA Driver API part */

CUresult CUDAAPI cuLaunchKernel(CUfunction f, 
                                unsigned int gridDimX, 
                                unsigned int gridDimY,
                                unsigned int gridDimZ, 
                                unsigned int blockDimX, 
                                unsigned int blockDimY,
                                unsigned int blockDimZ, 
                                unsigned int sharedMemBytes, 
                                CUstream hStream,
                                void **kernelParams, 
                                void **extra) {
  const char *symbolName = *(const char **)((uintptr_t) f + 8);
  DEBUG("Enter cuLaunchKernel(), symbol = %s\n", symbolName);

  typedef decltype(&cuLaunchKernel) funcType;
  funcType func = (funcType) actualDlsym(libcudaHandle, STRINGIFY(cuLaunchKernel));

  if (!inCudnnCublasInvocation) {
    fromCudaKernelParameters(kernelParams);
    fromCudaKernelSymbol(symbolName);
    prefetchPreKernel((cudaStream_t) hStream);
  }

#ifdef ANALYSIS
  if (inCudnnInvocation) {
    analysisDumpCudnnKernel();
  } else if (inCublasInvocation) {
    analysisDumpCublasKernel();
  } else {
    analysisDumpNativeKernel();
  }
#endif
  CUresult ret = func(f, 
                      gridDimX, 
                      gridDimY, 
                      gridDimZ, 
                      blockDimX, 
                      blockDimY, 
                      blockDimZ,
                      sharedMemBytes, 
                      hStream, 
                      kernelParams, 
                      extra);

  if (!inCudnnCublasInvocation) {
    prefetchPostKernel((cudaStream_t) hStream);
  }

  DEBUG("cuLaunchKernel() returns %d\n", (int) ret);

  return ret;
}

CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* dptr,
                                   size_t bytesize,
                                   unsigned int flags) {
  if (bytesize == 0) {
    return CUDA_SUCCESS;
  }

  DEBUG("Enter cuMemAllocManaged(), size = %llu\n", (unsigned long long int) bytesize);
  
  typedef decltype(&cuMemAllocManaged) funcType;
  funcType func = (funcType) actualDlsym(libcudaHandle, STRINGIFY(cuMemAllocManaged));

  CUresult ret = func(dptr,
                      bytesize,
                      flags);

  if (ret != CUDA_SUCCESS) {
    WARN("cuMemAllocManaged() returns %d\n", (int) ret);
  } else {
    DEBUG("cuMemAllocManaged(), dptr = %llx, size = %llu\n", (unsigned long long int) *dptr, (unsigned long long int) bytesize);
    fromCudaMalloc((void*) *dptr, bytesize);
  }

  return ret;
}

CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr,
                            size_t bytesize) {
  return cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
}

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (dptr == 0) {
    return CUDA_SUCCESS;
  }

  DEBUG("Enter cuMemFree(), dptr = %llx\n", (unsigned long long int) dptr);

  CUresult (*func)(CUdeviceptr);
  func = (CUresult(*)(CUdeviceptr)) actualDlsym(libcudaHandle, STRINGIFY(cuMemFree));

  CUresult ret = func(dptr);

  if (ret != CUDA_SUCCESS) {
    WARN("cuMemFree() returns %d\n", (int) ret);
  } else {
    DEBUG("Exit cuMemFree(), dptr = %llx\n", (unsigned long long int) dptr);
    fromCudaFree((void*) dptr);
  }

  return ret;
}
