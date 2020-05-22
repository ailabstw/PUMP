void prefetchPreKernel(cudaStream_t stream);
void prefetchPostKernel(cudaStream_t stream);
void pushToPrefetchCache(uint64_t ptr, uint64_t size);
void invalidateMemoryBlockInPrefetchCache(uint64_t ptr, uint64_t size);
