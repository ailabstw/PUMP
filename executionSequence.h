void executionSequenceCudnnCublasApi(const char *name);
void executionSequenceCudaKernel(const char *name);

void executionSequenceCudnnCublasAccessBlock(const void *ptr, const size_t size, bool fromCudnn);
void executionSequenceCudaAccessBlock(const void *ptr, const size_t size);
void executionSequenceFrameworkAccessBlock(const void *ptr, const size_t size);

void executionSequenceRemoveBlock(uint64_t ptr, uint64_t size);
