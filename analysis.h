#ifdef ANALYSIS
void analysisDumpCudnnCublasApiName(const char *name);
void analysisDumpCudaKernelName(const char *name);
void analysisDumpPrefetchCache(std::map<uint64_t, uint64_t> prefetchCache);
void analysisDumpEventQueueCount(std::map<std::pair<uint64_t, uint64_t>, uint64_t> eventQueueCount);
void analysisDumpMemoryBlockActivation(uint64_t ptr, uint64_t size);
void analysisDumpMemoryBlockDeactivation(uint64_t ptr, uint64_t size);
void analysisDumpMemoryBlockAllocation(uint64_t ptr, uint64_t size);
void analysisDumpMemoryBlockDeallocation(uint64_t ptr);
void analysisDumpCudaMalloc(uint64_t ptr, uint64_t size);
void analysisDumpCudnnKernel(void);
void analysisDumpCublasKernel(void);
void analysisDumpNativeKernel(void);
#endif
