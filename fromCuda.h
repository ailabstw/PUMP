void fromCudaKernelSymbol(const char *symbolName);
void fromCudaKernelParameters(void **kernelParams);

void fromCudaMalloc(void* ptr, size_t size);
void fromCudaFree(void* ptr);
