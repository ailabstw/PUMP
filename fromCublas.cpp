#include "global.h"
#include "executionSequence.h"

void fromCublasApiName(const char *apiName) {
  DEBUG("CUBLAS apiName = %s\n", apiName);
  executionSequenceCudnnCublasApi(apiName);
}

void fromCublasInoutPointerWithSize(void *pointer, size_t size) {
  executionSequenceCudnnCublasAccessBlock(pointer, size, false);
}
void fromCublasOutputPointerWithSize(void *pointer, size_t size) {
  executionSequenceCudnnCublasAccessBlock(pointer, size, false);
}
void fromCublasInputPointerWithSize(const void *pointer, size_t size) {
  executionSequenceCudnnCublasAccessBlock(pointer, size, false);
}
