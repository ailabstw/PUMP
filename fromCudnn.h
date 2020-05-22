void fromCudnnApiName(const char *apiName);

void fromCudnnCreateTensorDesc(const cudnnTensorDescriptor_t *tensorDesc);
void fromCudnnDestroyTensorDesc(const cudnnTensorDescriptor_t tensorDesc);
void fromCudnnSetTensorDescSize(const cudnnTensorDescriptor_t tensorDesc);

void fromCudnnInputTensor(const cudnnTensorDescriptor_t desc, const void *tensor);
void fromCudnnOutputTensor(const cudnnTensorDescriptor_t desc, const void *tensor);
void fromCudnnInputOutputTensor(const cudnnTensorDescriptor_t desc, const void *tensor);

void fromCudnnInputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length);
void fromCudnnOutputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length);
void fromCudnnInputOutputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length);

void fromCudnnWorkspace(const void *workspace, size_t bytes);

void fromCudnnCreateFilterDesc(const cudnnFilterDescriptor_t *filterDesc);
void fromCudnnDestroyFilterDesc(const cudnnFilterDescriptor_t filterDesc);
void fromCudnnSetFilterDescSize(const cudnnFilterDescriptor_t filterDesc);
void fromCudnnSetFilter4dDesc(const cudnnFilterDescriptor_t filterDesc,
                         const cudnnDataType_t dataType,
                         const cudnnTensorFormat_t format,
                         const int k,
                         const int c,
                         const int h,
                         const int w);
void fromCudnnSetFilterNdDesc(const cudnnFilterDescriptor_t filterDesc,
                         const cudnnDataType_t dataType,
                         const cudnnTensorFormat_t format,
                         const int nbDims,
                         const int filterDimA[]);

void fromCudnnInputFilter(const cudnnFilterDescriptor_t desc, const void *filter);
void fromCudnnOutputFilter(const cudnnFilterDescriptor_t desc, const void *filter);
void fromCudnnInputOutputFilter(const cudnnFilterDescriptor_t desc, const void *filter);
