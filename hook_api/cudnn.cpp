#define CUDNN_ENTER_API(name)                                                  \
  static bool warned = false;                                                  \
  if (!warned) {                                                               \
    WARN("Enter " STRINGIFY(name) "(). Optimization may be required.\n");      \
    warned = true;                                                             \
  }                                                                            \
  DEBUG("Enter " STRINGIFY(name) "(). Optimization may be required.\n");       \
  typedef decltype(&name) funcType;                                            \
  funcType func = (funcType) actualDlsym(libcudnnHandle, STRINGIFY(name));     \

#define CUDNN_ENTER_API_OPTIMIZED(name)                                        \
  DEBUG("Enter " STRINGIFY(name) "()\n");                                      \
  typedef decltype(&name) funcType;                                            \
  funcType func = (funcType) actualDlsym(libcudnnHandle, STRINGIFY(name));     \

#define CUDNN_ENTER_COMPUTE_KERNEL(name, handle)                               \
  fromCudnnApiName(STRINGIFY(name));                                           \
  cudaStream_t stream;                                                         \
  cudnnGetStream(handle, &stream);                                             \
  prefetchPreKernel(stream);                                                   \
  if (inCudnnCublasInvocation) {                                               \
    ERROR("ERROR\n");                                                          \
  }                                                                            \
  inCudnnCublasInvocation = true;                                              \
  inCudnnInvocation = true;                                                    \

#define CUDNN_LEAVE_COMPUTE_KERNEL(handle)                                     \
  inCudnnCublasInvocation = false;                                             \
  inCudnnInvocation = false;                                                   \
  prefetchPostKernel(stream);                                                  \

size_t cudnnGetVersion() {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetVersion);
  int ret = func();
  return ret;
}

size_t cudnnGetCudartVersion() {
  CUDNN_ENTER_API(cudnnGetCudartVersion);
  int ret = func();
  return ret;
}

const char *cudnnGetErrorString(cudnnStatus_t status) {
  CUDNN_ENTER_API(cudnnGetErrorString);
  const char *ret = func(status);
  return ret;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t handle,
                                     cudnnStatus_t *rstatus,
                                     cudnnErrQueryMode_t mode,
                                     cudnnRuntimeTag_t *tag) {
  CUDNN_ENTER_API(cudnnQueryRuntimeError);
  cudnnStatus_t ret = func(handle, rstatus, mode, tag);
  return ret;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int *value) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetProperty);
  cudnnStatus_t ret = func(type, value);
  return ret;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreate);
  cudnnStatus_t ret = func(handle);
  return ret;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroy);
  cudnnStatus_t ret = func(handle);
  return ret;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetStream);
  cudnnStatus_t ret = func(handle, streamId);
  return ret;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetStream);
  cudnnStatus_t ret = func(handle, streamId);
  return ret;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreateTensorDescriptor);

  cudnnStatus_t ret = func(tensorDesc);
  fromCudnnCreateTensorDesc(tensorDesc);
  return ret;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnTensorFormat_t format,
                                         cudnnDataType_t dataType, int n, int c,
                                         int h, int w) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetTensor4dDescriptor);
  cudnnStatus_t ret = func(tensorDesc, format, dataType, n, c, h, w);

  fromCudnnSetTensorDescSize(tensorDesc);

  return ret;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                           cudnnDataType_t dataType, int n,
                                           int c, int h, int w, int nStride,
                                           int cStride, int hStride,
                                           int wStride) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetTensor4dDescriptorEx);
  cudnnStatus_t ret = func(tensorDesc, dataType, n, c, h, w, nStride, cStride,
                           hStride, wStride);

  fromCudnnSetTensorDescSize(tensorDesc);

  return ret;
}

cudnnStatus_t
cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t *dataType, int *n, int *c, int *h,
                           int *w, int *nStride, int *cStride, int *hStride,
                           int *wStride) {
  CUDNN_ENTER_API(cudnnGetTensor4dDescriptor);
  cudnnStatus_t ret = func(tensorDesc, dataType, n, c, h, w, nStride, cStride,
                           hStride, wStride);
  return ret;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                         cudnnDataType_t dataType, int nbDims,
                                         const int dimA[],
                                         const int strideA[]) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetTensorNdDescriptor);
  cudnnStatus_t ret = func(tensorDesc, dataType, nbDims, dimA, strideA);

  fromCudnnSetTensorDescSize(tensorDesc);

  return ret;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                           cudnnTensorFormat_t format,
                                           cudnnDataType_t dataType, int nbDims,
                                           const int dimA[]) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetTensorNdDescriptorEx);
  cudnnStatus_t ret = func(tensorDesc, format, dataType, nbDims, dimA);

  fromCudnnSetTensorDescSize(tensorDesc);

  return ret;
}

cudnnStatus_t
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested, cudnnDataType_t *dataType,
                           int *nbDims, int dimA[], int strideA[]) {
  CUDNN_ENTER_API(cudnnGetTensorNdDescriptor);
  cudnnStatus_t ret =
      func(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
  return ret;
}

cudnnStatus_t
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc,
                          size_t *size) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetTensorSizeInBytes);
  cudnnStatus_t ret = func(tensorDesc, size);
  return ret;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroyTensorDescriptor);
  fromCudnnDestroyTensorDesc(tensorDesc);

  cudnnStatus_t ret = func(tensorDesc);
  return ret;
}

cudnnStatus_t
cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                       const cudnnTensorDescriptor_t srcDesc,
                       cudnnTensorDescriptor_t destDesc,
                       size_t *destSizeInBytes) {
  CUDNN_ENTER_API(cudnnInitTransformDest);
  cudnnStatus_t ret = func(transformDesc, srcDesc, destDesc, destSizeInBytes);
  return ret;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims,
    const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[],
    const int32_t padAfterA[], const uint32_t foldA[],
    const cudnnFoldingDirection_t direction) {
  CUDNN_ENTER_API(cudnnSetTensorTransformDescriptor);
  cudnnStatus_t ret = func(transformDesc, nbDims, destFormat, padBeforeA,
                           padAfterA, foldA, direction);
  return ret;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested,
    cudnnTensorFormat_t *destFormat, int32_t padBeforeA[], int32_t padAfterA[],
    uint32_t foldA[], cudnnFoldingDirection_t *direction) {
  CUDNN_ENTER_API(cudnnGetTensorTransformDescriptor);
  cudnnStatus_t ret = func(transformDesc, nbDimsRequested, destFormat,
                           padBeforeA, padAfterA, foldA, direction);
  return ret;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc) {
  CUDNN_ENTER_API(cudnnDestroyTensorTransformDescriptor);

  cudnnStatus_t ret = func(transformDesc);
  return ret;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc,
                                   const void *x, const void *beta,
                                   const cudnnTensorDescriptor_t yDesc,
                                   void *y) {
  CUDNN_ENTER_API(cudnnTransformTensor);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnTransformTensor, handle);
  cudnnStatus_t ret = func(handle, alpha, xDesc, x, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnTransformTensorEx(cudnnHandle_t handle,
                       const cudnnTensorTransformDescriptor_t transDesc,
                       const void *alpha, const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData, const void *beta,
                       const cudnnTensorDescriptor_t destDesc, void *destData) {
  CUDNN_ENTER_API(cudnnTransformTensorEx);
  fromCudnnInputTensor(srcDesc, srcData);
  fromCudnnOutputTensor(destDesc, destData);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnTransformTensorEx, handle);
  cudnnStatus_t ret = func(handle, transDesc, alpha, srcDesc, srcData, beta,
                           destDesc, destData);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat,
    cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc,
    cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc,
    cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc,
    cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
  CUDNN_ENTER_API(cudnnGetFoldedConvBackwardDataDescriptors);
  cudnnStatus_t ret =
      func(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat,
           foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc,
           filterFoldTransDesc, diffPadTransDesc, gradFoldTransDesc,
           gradUnfoldTransDesc);
  return ret;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                             const cudnnTensorDescriptor_t aDesc, const void *A,
                             const void *beta,
                             const cudnnTensorDescriptor_t cDesc, void *C) {
  CUDNN_ENTER_API(cudnnAddTensor);
  fromCudnnInputTensor(aDesc, A);
  fromCudnnOutputTensor(cDesc, C);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnAddTensor, handle);
  cudnnStatus_t ret = func(handle, alpha, aDesc, A, beta, cDesc, C);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
  CUDNN_ENTER_API(cudnnCreateOpTensorDescriptor);
  cudnnStatus_t ret = func(opTensorDesc);
  return ret;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                         cudnnOpTensorOp_t opTensorOp,
                                         cudnnDataType_t opTensorCompType,
                                         cudnnNanPropagation_t opTensorNanOpt) {
  CUDNN_ENTER_API(cudnnSetOpTensorDescriptor);
  cudnnStatus_t ret =
      func(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  return ret;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(
    const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp,
    cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt) {
  CUDNN_ENTER_API(cudnnGetOpTensorDescriptor);
  cudnnStatus_t ret =
      func(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
  return ret;
}

cudnnStatus_t
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
  CUDNN_ENTER_API(cudnnDestroyOpTensorDescriptor);
  cudnnStatus_t ret = func(opTensorDesc);
  return ret;
}

cudnnStatus_t cudnnOpTensor(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
  CUDNN_ENTER_API(cudnnOpTensor);
  fromCudnnInputTensor(aDesc, A);
  fromCudnnInputTensor(bDesc, B);
  fromCudnnOutputTensor(cDesc, C);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnOpTensor, handle);
  cudnnStatus_t ret = func(handle, opTensorDesc, alpha1, aDesc, A, alpha2,
                           bDesc, B, beta, cDesc, C);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
  CUDNN_ENTER_API(cudnnCreateReduceTensorDescriptor);
  cudnnStatus_t ret = func(reduceTensorDesc);
  return ret;
}

cudnnStatus_t
cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t reduceTensorOp,
                               cudnnDataType_t reduceTensorCompType,
                               cudnnNanPropagation_t reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t reduceTensorIndices,
                               cudnnIndicesType_t reduceTensorIndicesType) {
  CUDNN_ENTER_API(cudnnSetReduceTensorDescriptor);
  cudnnStatus_t ret =
      func(reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
           reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
  return ret;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType,
    cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices,
    cudnnIndicesType_t *reduceTensorIndicesType) {
  CUDNN_ENTER_API(cudnnGetReduceTensorDescriptor);
  cudnnStatus_t ret =
      func(reduceTensorDesc, reduceTensorOp, reduceTensorCompType,
           reduceTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
  return ret;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(
    cudnnReduceTensorDescriptor_t reduceTensorDesc) {
  CUDNN_ENTER_API(cudnnDestroyReduceTensorDescriptor);
  cudnnStatus_t ret = func(reduceTensorDesc);
  return ret;
}

cudnnStatus_t cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  CUDNN_ENTER_API(cudnnGetReductionIndicesSize);
  cudnnStatus_t ret = func(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
  CUDNN_ENTER_API(cudnnGetReductionWorkspaceSize);
  cudnnStatus_t ret = func(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnReduceTensor(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc,
    void *indices, size_t indicesSizeInBytes, void *workspace,
    size_t workspaceSizeInBytes, const void *alpha,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta,
    const cudnnTensorDescriptor_t cDesc, void *C) {
  CUDNN_ENTER_API(cudnnReduceTensor);
  fromCudnnWorkspace(indices, indicesSizeInBytes);
  fromCudnnWorkspace(workspace, workspaceSizeInBytes);
  fromCudnnInputTensor(aDesc, A);
  fromCudnnOutputTensor(cDesc, C);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnReduceTensor, handle);
  cudnnStatus_t ret =
      func(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace,
           workspaceSizeInBytes, alpha, aDesc, A, beta, cDesc, C);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t handle,
                             const cudnnTensorDescriptor_t yDesc, void *y,
                             const void *valuePtr) {
  CUDNN_ENTER_API(cudnnSetTensor);
  fromCudnnOutputTensor(yDesc, y);
  cudnnStatus_t ret = func(handle, yDesc, y, valuePtr);
  return ret;
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t handle,
                               const cudnnTensorDescriptor_t yDesc, void *y,
                               const void *alpha) {
  CUDNN_ENTER_API(cudnnScaleTensor);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnScaleTensor, handle);
  cudnnStatus_t ret = func(handle, yDesc, y, alpha);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreateFilterDescriptor);

  cudnnStatus_t ret = func(filterDesc);
  fromCudnnCreateFilterDesc(filterDesc);
  return ret;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int k,
                                         int c, int h, int w) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetFilter4dDescriptor);
  cudnnStatus_t ret = func(filterDesc, dataType, format, k, c, h, w);

  fromCudnnSetFilter4dDesc(filterDesc,
                      dataType,
                      format,
                      k,
                      c,
                      h,
                      w);

  return ret;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(
    const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType,
    cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w) {
  CUDNN_ENTER_API(cudnnGetFilter4dDescriptor);
  cudnnStatus_t ret = func(filterDesc, dataType, format, k, c, h, w);
  return ret;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format, int nbDims,
                                         const int filterDimA[]) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetFilterNdDescriptor);
  cudnnStatus_t ret = func(filterDesc, dataType, format, nbDims, filterDimA);

  fromCudnnSetFilterNdDesc(filterDesc,
                      dataType,
                      format,
                      nbDims,
                      filterDimA);

  return ret;
}

cudnnStatus_t
cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           int nbDimsRequested, cudnnDataType_t *dataType,
                           cudnnTensorFormat_t *format, int *nbDims,
                           int filterDimA[]) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetFilterNdDescriptor);
  cudnnStatus_t ret =
      func(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
  return ret;
}

cudnnStatus_t
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc,
                          size_t *size) {
  CUDNN_ENTER_API(cudnnGetFilterSizeInBytes);
  cudnnStatus_t ret = func(filterDesc, size);
  return ret;
}

cudnnStatus_t
cudnnTransformFilter(cudnnHandle_t handle,
                     const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha, const cudnnFilterDescriptor_t srcDesc,
                     const void *srcData, const void *beta,
                     const cudnnFilterDescriptor_t destDesc, void *destData) {
  CUDNN_ENTER_API(cudnnTransformFilter);
  fromCudnnInputFilter(srcDesc, srcData);
  fromCudnnOutputFilter(destDesc, destData);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnTransformFilter, handle);
  cudnnStatus_t ret = func(handle, transDesc, alpha, srcDesc, srcData, beta,
                           destDesc, destData);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroyFilterDescriptor);
  fromCudnnDestroyFilterDesc(filterDesc);

  cudnnStatus_t ret = func(filterDesc);
  return ret;
}

cudnnStatus_t cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    cudnnReorderType_t reorderType, const void *filterData,
    void *reorderedFilterData, int reorderBias, const void *biasData,
    void *reorderedBiasData) {
  CUDNN_ENTER_API(cudnnReorderFilterAndBias);
  cudnnStatus_t ret =
      func(handle, filterDesc, reorderType, filterData, reorderedFilterData,
           reorderBias, biasData, reorderedBiasData);
  return ret;
}

cudnnStatus_t
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreateConvolutionDescriptor);
  cudnnStatus_t ret = func(convDesc);
  return ret;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t mathType) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetConvolutionMathType);
  cudnnStatus_t ret = func(convDesc, mathType);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                          cudnnMathType_t *mathType) {
  CUDNN_ENTER_API(cudnnGetConvolutionMathType);
  cudnnStatus_t ret = func(convDesc, mathType);
  return ret;
}

cudnnStatus_t
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int groupCount) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetConvolutionGroupCount);
  cudnnStatus_t ret = func(convDesc, groupCount);
  return ret;
}

cudnnStatus_t
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                              int *groupCount) {
  CUDNN_ENTER_API(cudnnGetConvolutionGroupCount);
  cudnnStatus_t ret = func(convDesc, groupCount);
  return ret;
}

cudnnStatus_t
cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                               cudnnReorderType_t reorderType) {
  CUDNN_ENTER_API(cudnnSetConvolutionReorderType);
  cudnnStatus_t ret = func(convDesc, reorderType);
  return ret;
}

cudnnStatus_t
cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                               cudnnReorderType_t *reorderType) {
  CUDNN_ENTER_API(cudnnGetConvolutionReorderType);
  cudnnStatus_t ret = func(convDesc, reorderType);
  return ret;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int dilation_h, int dilation_w, cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType) {
  CUDNN_ENTER_API(cudnnSetConvolution2dDescriptor);
  cudnnStatus_t ret = func(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w,
                           mode, computeType);
  return ret;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u,
    int *v, int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode,
    cudnnDataType_t *computeType) {
  CUDNN_ENTER_API(cudnnGetConvolution2dDescriptor);
  cudnnStatus_t ret = func(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w,
                           mode, computeType);
  return ret;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
  CUDNN_ENTER_API(cudnnGetConvolution2dForwardOutputDim);
  cudnnStatus_t ret = func(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
  return ret;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[],
    const int filterStrideA[], const int dilationA[],
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetConvolutionNdDescriptor);
  cudnnStatus_t ret = func(convDesc, arrayLength, padA, filterStrideA,
                           dilationA, mode, computeType);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(
    const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested,
    int *arrayLength, int padA[], int strideA[], int dilationA[],
    cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType) {
  CUDNN_ENTER_API(cudnnGetConvolutionNdDescriptor);
  cudnnStatus_t ret = func(convDesc, arrayLengthRequested, arrayLength, padA,
                           strideA, dilationA, mode, computeType);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims,
    int tensorOuputDimA[]) {
  CUDNN_ENTER_API(cudnnGetConvolutionNdForwardOutputDim);
  cudnnStatus_t ret =
      func(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
  return ret;
}

cudnnStatus_t
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroyConvolutionDescriptor);
  cudnnStatus_t ret = func(convDesc);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
                                                          int *count) {
  CUDNN_ENTER_API(cudnnGetConvolutionForwardAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, count);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API(cudnnFindConvolutionForwardAlgorithm);
  cudnnStatus_t ret = func(handle, xDesc, wDesc, convDesc, yDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnFindConvolutionForwardAlgorithmEx);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(yDesc, y);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindConvolutionForwardAlgorithmEx, handle);
  cudnnStatus_t ret =
      func(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount,
           returnedAlgoCount, perfResults, workSpace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionFwdAlgo_t *algo) {
  CUDNN_ENTER_API(cudnnGetConvolutionForwardAlgorithm);
  cudnnStatus_t ret = func(handle, xDesc, wDesc, convDesc, yDesc, preference,
                           memoryLimitInBytes, algo);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionForwardAlgorithm_v7);
  cudnnStatus_t ret = func(handle, srcDesc, filterDesc, convDesc, destDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionForwardWorkspaceSize);
  cudnnStatus_t ret =
      func(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  return ret;
}

cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnConvolutionForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnConvolutionForward, handle);
  cudnnStatus_t ret = func(handle, alpha, xDesc, x, wDesc, w, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnConvolutionBiasActivationForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnInputTensor(zDesc, z);
  fromCudnnInputTensor(biasDesc, bias);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnConvolutionBiasActivationForward, handle);
  cudnnStatus_t ret = func(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, alpha2, zDesc, z,
                           biasDesc, bias, activationDesc, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                           const void *alpha,
                                           const cudnnTensorDescriptor_t dyDesc,
                                           const void *dy, const void *beta,
                                           const cudnnTensorDescriptor_t dbDesc,
                                           void *db) {
  CUDNN_ENTER_API(cudnnConvolutionBackwardBias);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnOutputTensor(dbDesc, db);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnConvolutionBackwardBias, handle);
  cudnnStatus_t ret = func(handle, alpha, dyDesc, dy, beta, dbDesc, db);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle,
                                                   int *count) {
  CUDNN_ENTER_API(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, count);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API(cudnnFindConvolutionBackwardFilterAlgorithm);
  cudnnStatus_t ret = func(handle, xDesc, dyDesc, convDesc, dwDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *y,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnFindConvolutionBackwardFilterAlgorithmEx);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputTensor(dyDesc, y);
  fromCudnnOutputFilter(dwDesc, dw);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindConvolutionBackwardFilterAlgorithmEx, handle);
  cudnnStatus_t ret = func(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw,
                           requestedAlgoCount, returnedAlgoCount, perfResults,
                           workSpace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
  CUDNN_ENTER_API(cudnnGetConvolutionBackwardFilterAlgorithm);
  cudnnStatus_t ret = func(handle, xDesc, dyDesc, convDesc, dwDesc, preference,
                           memoryLimitInBytes, algo);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
  cudnnStatus_t ret = func(handle, srcDesc, diffDesc, convDesc, gradDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionBackwardFilterWorkspaceSize);
  cudnnStatus_t ret =
      func(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnConvolutionBackwardFilter);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnOutputFilter(dwDesc, dw);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnConvolutionBackwardFilter, handle);
  cudnnStatus_t ret = func(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                 int *count) {
  CUDNN_ENTER_API(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, count);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API(cudnnFindConvolutionBackwardDataAlgorithm);
  cudnnStatus_t ret = func(handle, wDesc, dyDesc, convDesc, dxDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnFindConvolutionBackwardDataAlgorithmEx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnOutputTensor(dxDesc, dx);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindConvolutionBackwardDataAlgorithmEx, handle);
  cudnnStatus_t ret = func(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
                           requestedAlgoCount, returnedAlgoCount, perfResults,
                           workSpace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdDataAlgo_t *algo) {
  CUDNN_ENTER_API(cudnnGetConvolutionBackwardDataAlgorithm);
  cudnnStatus_t ret = func(handle, wDesc, dyDesc, convDesc, dxDesc, preference,
                           memoryLimitInBytes, algo);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc,
    const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionBackwardDataAlgorithm_v7);
  cudnnStatus_t ret = func(handle, filterDesc, diffDesc, convDesc, gradDesc,
                           requestedAlgoCount, returnedAlgoCount, perfResults);
  return ret;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc,
    const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetConvolutionBackwardDataWorkspaceSize);
  cudnnStatus_t ret =
      func(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnConvolutionBackwardData);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnOutputTensor(dxDesc, dx);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnConvolutionBackwardData, handle);
  cudnnStatus_t ret = func(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo,
                           workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnIm2Col(cudnnHandle_t handle,
                          const cudnnTensorDescriptor_t xDesc, const void *x,
                          const cudnnFilterDescriptor_t wDesc,
                          const cudnnConvolutionDescriptor_t convDesc,
                          void *colBuffer) {
  CUDNN_ENTER_API(cudnnIm2Col);
  fromCudnnInputTensor(xDesc, x);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnIm2Col, handle);
  cudnnStatus_t ret = func(handle, xDesc, x, wDesc, convDesc, colBuffer);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnSoftmaxForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnSoftmaxForward, handle);
  cudnnStatus_t ret = func(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSoftmaxBackward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_ENTER_API(cudnnSoftmaxBackward);
  fromCudnnInputTensor(yDesc, y);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnOutputTensor(dxDesc, dx);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnSoftmaxBackward, handle);
  cudnnStatus_t ret =
      func(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
  CUDNN_ENTER_API(cudnnCreatePoolingDescriptor);
  cudnnStatus_t ret = func(poolingDesc);
  return ret;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode,
    cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride) {
  CUDNN_ENTER_API(cudnnSetPooling2dDescriptor);
  cudnnStatus_t ret = func(poolingDesc, mode, maxpoolingNanOpt, windowHeight,
                           windowWidth, verticalPadding, horizontalPadding,
                           verticalStride, horizontalStride);
  return ret;
}

cudnnStatus_t cudnnGetPooling2dDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode,
    cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
  CUDNN_ENTER_API(cudnnGetPooling2dDescriptor);
  cudnnStatus_t ret = func(poolingDesc, mode, maxpoolingNanOpt, windowHeight,
                           windowWidth, verticalPadding, horizontalPadding,
                           verticalStride, horizontalStride);
  return ret;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(
    cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode,
    const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]) {
  CUDNN_ENTER_API(cudnnSetPoolingNdDescriptor);
  cudnnStatus_t ret = func(poolingDesc, mode, maxpoolingNanOpt, nbDims,
                           windowDimA, paddingA, strideA);
  return ret;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(
    const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested,
    cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt,
    int *nbDims, int windowDimA[], int paddingA[], int strideA[]) {
  CUDNN_ENTER_API(cudnnGetPoolingNdDescriptor);
  cudnnStatus_t ret = func(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt,
                           nbDims, windowDimA, paddingA, strideA);
  return ret;
}

cudnnStatus_t
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims, int outputTensorDimA[]) {
  CUDNN_ENTER_API(cudnnGetPoolingNdForwardOutputDim);
  cudnnStatus_t ret =
      func(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
  return ret;
}

cudnnStatus_t
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n, int *c, int *h, int *w) {
  CUDNN_ENTER_API(cudnnGetPooling2dForwardOutputDim);
  cudnnStatus_t ret = func(poolingDesc, inputTensorDesc, n, c, h, w);
  return ret;
}

cudnnStatus_t
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
  CUDNN_ENTER_API(cudnnDestroyPoolingDescriptor);
  cudnnStatus_t ret = func(poolingDesc);
  return ret;
}

cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnPoolingForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnPoolingForward, handle);
  cudnnStatus_t ret =
      func(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnPoolingBackward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_ENTER_API(cudnnPoolingBackward);
  fromCudnnInputTensor(yDesc, y);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(dxDesc, dx);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnPoolingBackward, handle);
  cudnnStatus_t ret = func(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy,
                           xDesc, x, beta, dxDesc, dx);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
  CUDNN_ENTER_API(cudnnCreateActivationDescriptor);
  cudnnStatus_t ret = func(activationDesc);
  return ret;
}

cudnnStatus_t
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt, double coef) {
  CUDNN_ENTER_API(cudnnSetActivationDescriptor);
  cudnnStatus_t ret = func(activationDesc, mode, reluNanOpt, coef);
  return ret;
}

cudnnStatus_t
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt, double *coef) {
  CUDNN_ENTER_API(cudnnGetActivationDescriptor);
  cudnnStatus_t ret = func(activationDesc, mode, reluNanOpt, coef);
  return ret;
}

cudnnStatus_t
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
  CUDNN_ENTER_API(cudnnDestroyActivationDescriptor);
  cudnnStatus_t ret = func(activationDesc);
  return ret;
}

cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnActivationForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnActivationForward, handle);
  cudnnStatus_t ret =
      func(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_ENTER_API(cudnnActivationBackward);
  fromCudnnInputTensor(yDesc, y);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(dxDesc, dx);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnActivationBackward, handle);
  cudnnStatus_t ret = func(handle, activationDesc, alpha, yDesc, y, dyDesc, dy,
                           xDesc, x, beta, dxDesc, dx);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
  CUDNN_ENTER_API(cudnnCreateLRNDescriptor);
  cudnnStatus_t ret = func(normDesc);
  return ret;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                    unsigned lrnN, double lrnAlpha,
                                    double lrnBeta, double lrnK) {
  CUDNN_ENTER_API(cudnnSetLRNDescriptor);
  cudnnStatus_t ret = func(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  return ret;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
                                    unsigned *lrnN, double *lrnAlpha,
                                    double *lrnBeta, double *lrnK) {
  CUDNN_ENTER_API(cudnnGetLRNDescriptor);
  cudnnStatus_t ret = func(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
  return ret;
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
  CUDNN_ENTER_API(cudnnDestroyLRNDescriptor);
  cudnnStatus_t ret = func(lrnDesc);
  return ret;
}

cudnnStatus_t cudnnLRNCrossChannelForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnLRNCrossChannelForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnLRNCrossChannelForward, handle);
  cudnnStatus_t ret =
      func(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  CUDNN_ENTER_API(cudnnLRNCrossChannelBackward);
  fromCudnnInputTensor(yDesc, y);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(dxDesc, dx);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnLRNCrossChannelBackward, handle);
  cudnnStatus_t ret = func(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc,
                           dy, xDesc, x, beta, dxDesc, dx);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means,
    void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_ENTER_API(cudnnDivisiveNormalizationForward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputTensor(xDesc, means);
  fromCudnnOutputTensor(xDesc, temp);
  fromCudnnOutputTensor(xDesc, temp2);
  fromCudnnOutputTensor(yDesc, y);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnDivisiveNormalizationForward, handle);
  cudnnStatus_t ret = func(handle, normDesc, mode, alpha, xDesc, x, means, temp,
                           temp2, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc,
    cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means,
    const void *dy, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t dXdMeansDesc, void *dx, void *dMeans) {
  CUDNN_ENTER_API(cudnnDivisiveNormalizationBackward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputTensor(xDesc, means);
  fromCudnnOutputTensor(xDesc, temp);
  fromCudnnOutputTensor(xDesc, temp2);
  fromCudnnOutputTensor(dXdMeansDesc, dx);
  fromCudnnOutputTensor(dXdMeansDesc, dMeans);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnDivisiveNormalizationBackward, handle);
  cudnnStatus_t ret = func(handle, normDesc, mode, alpha, xDesc, x, means, dy,
                           temp, temp2, beta, dXdMeansDesc, dx, dMeans);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc,
                              cudnnBatchNormMode_t mode) {
  CUDNN_ENTER_API(cudnnDeriveBNTensorDescriptor);
  cudnnStatus_t ret = func(derivedBnDesc, xDesc, mode);
  return ret;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc,
    const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
  cudnnStatus_t ret = func(handle, mode, bnOps, xDesc, zDesc, yDesc,
                           bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
  cudnnStatus_t ret =
      func(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc,
           dBnScaleBiasDesc, activationDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
  cudnnStatus_t ret =
      func(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance) {
  CUDNN_ENTER_API(cudnnBatchNormalizationForwardTraining);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnScale);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnBias);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnBatchNormalizationForwardTraining, handle);
  cudnnStatus_t ret = func(
      handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
      bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
      resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc,
    const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, double exponentialAverageFactor,
    void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnBatchNormalizationForwardTrainingEx);
  fromCudnnInputTensor(xDesc, xData);
  fromCudnnInputTensor(zDesc, zData);
  fromCudnnOutputTensor(yDesc, yData);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnScale);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnBias);
  fromCudnnInputOutputTensor(bnScaleBiasMeanVarDesc, resultRunningMean);
  fromCudnnInputOutputTensor(bnScaleBiasMeanVarDesc, resultRunningVariance);
  fromCudnnOutputTensor(bnScaleBiasMeanVarDesc, resultSaveMean);
  fromCudnnOutputTensor(bnScaleBiasMeanVarDesc, resultSaveInvVariance);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnBatchNormalizationForwardTrainingEx, handle);
  cudnnStatus_t ret = func(
      handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc,
      yData, bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
      resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
      resultSaveInvVariance, activationDesc, workspace, workSpaceSizeInBytes,
      reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha,
    const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
  CUDNN_ENTER_API(cudnnBatchNormalizationForwardInference);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(yDesc, y);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnScale);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, bnBias);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, estimatedMean);
  fromCudnnInputTensor(bnScaleBiasMeanVarDesc, estimatedVariance);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnBatchNormalizationForwardInference, handle);
  cudnnStatus_t ret = func(handle, mode, alpha, beta, xDesc, x, yDesc, y,
                           bnScaleBiasMeanVarDesc, bnScale, bnBias,
                           estimatedMean, estimatedVariance, epsilon);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff,
    const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc,
    const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale,
    void *dBnScaleResult, void *dBnBiasResult, double epsilon,
    const void *savedMean, const void *savedInvVariance) {
  CUDNN_ENTER_API(cudnnBatchNormalizationBackward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnInputTensor(dyDesc, dy);
  fromCudnnOutputTensor(dxDesc, dx);
  fromCudnnInputTensor(dBnScaleBiasDesc, bnScale);
  fromCudnnOutputTensor(dBnScaleBiasDesc, dBnScaleResult);
  fromCudnnOutputTensor(dBnScaleBiasDesc, dBnBiasResult);
  fromCudnnOutputTensor(dBnScaleBiasDesc, savedMean);
  fromCudnnOutputTensor(dBnScaleBiasDesc, savedInvVariance);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnBatchNormalizationBackward, handle);
  cudnnStatus_t ret = func(
      handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
      xDesc, x, dyDesc, dy, dxDesc, dx, dBnScaleBiasDesc, bnScale,
      dBnScaleResult, dBnBiasResult, epsilon, savedMean, savedInvVariance);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData,
    const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData,
    const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScaleData,
    const void *bnBiasData, void *dBnScaleData, void *dBnBiasData,
    double epsilon, const void *savedMean, const void *savedInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnBatchNormalizationBackwardEx);
  fromCudnnInputTensor(xDesc, xData);
  fromCudnnInputTensor(yDesc, yData);
  fromCudnnInputTensor(dyDesc, dyData);
  fromCudnnOutputTensor(dzDesc, dzData);
  fromCudnnOutputTensor(dxDesc, dxData);
  fromCudnnInputTensor(dBnScaleBiasDesc, bnScaleData);
  fromCudnnInputTensor(dBnScaleBiasDesc, bnBiasData);
  fromCudnnOutputTensor(dBnScaleBiasDesc, dBnScaleData);
  fromCudnnOutputTensor(dBnScaleBiasDesc, dBnBiasData);
  fromCudnnOutputTensor(dBnScaleBiasDesc, savedMean);
  fromCudnnOutputTensor(dBnScaleBiasDesc, savedInvVariance);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnBatchNormalizationBackwardEx, handle);
  cudnnStatus_t ret = func(
      handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff,
      betaParamDiff, xDesc, xData, yDesc, yData, dyDesc, dyData, dzDesc, dzData,
      dxDesc, dxData, dBnScaleBiasDesc, bnScaleData, bnBiasData, dBnScaleData,
      dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc,
      workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t *stDesc) {
  CUDNN_ENTER_API(cudnnCreateSpatialTransformerDescriptor);
  cudnnStatus_t ret = func(stDesc);
  return ret;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
    cudnnDataType_t dataType, const int nbDims, const int dimA[]) {
  CUDNN_ENTER_API(cudnnSetSpatialTransformerNdDescriptor);
  cudnnStatus_t ret = func(stDesc, samplerType, dataType, nbDims, dimA);
  return ret;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(
    cudnnSpatialTransformerDescriptor_t stDesc) {
  CUDNN_ENTER_API(cudnnDestroySpatialTransformerDescriptor);
  cudnnStatus_t ret = func(stDesc);
  return ret;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *theta, void *grid) {
  CUDNN_ENTER_API(cudnnSpatialTfGridGeneratorForward);
  cudnnStatus_t ret = func(handle, stDesc, theta, grid);
  return ret;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc,
    const void *dgrid, void *dtheta) {
  CUDNN_ENTER_API(cudnnSpatialTfGridGeneratorBackward);
  cudnnStatus_t ret = func(handle, stDesc, dgrid, dtheta);
  return ret;
}

cudnnStatus_t cudnnSpatialTfSamplerForward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc,
    void *y) {
  CUDNN_ENTER_API(cudnnSpatialTfSamplerForward);
  fromCudnnInputTensor(xDesc, x);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnSpatialTfSamplerForward, handle);
  cudnnStatus_t ret =
      func(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const void *grid, const void *betaDgrid, void *dgrid) {
  CUDNN_ENTER_API(cudnnSpatialTfSamplerBackward);
  fromCudnnInputTensor(xDesc, x);
  fromCudnnOutputTensor(dxDesc, dx);
  fromCudnnInputTensor(dyDesc, dy);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnSpatialTfSamplerBackward, handle);
  cudnnStatus_t ret = func(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx,
                           alphaDgrid, dyDesc, dy, grid, betaDgrid, dgrid);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreateDropoutDescriptor);
  cudnnStatus_t ret = func(dropoutDesc);
  return ret;
}

cudnnStatus_t
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroyDropoutDescriptor);
  cudnnStatus_t ret = func(dropoutDesc);
  return ret;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle,
                                        size_t *sizeInBytes) {
  CUDNN_ENTER_API(cudnnDropoutGetStatesSize);
  cudnnStatus_t ret = func(handle, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                              size_t *sizeInBytes) {
  CUDNN_ENTER_API(cudnnDropoutGetReserveSpaceSize);
  cudnnStatus_t ret = func(xdesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnHandle_t handle, float dropout,
                                        void *states, size_t stateSizeInBytes,
                                        unsigned long long seed) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetDropoutDescriptor);
  fromCudnnWorkspace(states, stateSizeInBytes);
  cudnnStatus_t ret =
      func(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  return ret;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(
    cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout,
    void *states, size_t stateSizeInBytes, unsigned long long seed) {
  CUDNN_ENTER_API(cudnnRestoreDropoutDescriptor);
  fromCudnnWorkspace(states, stateSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRestoreDropoutDescriptor, handle);
  cudnnStatus_t ret =
      func(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnHandle_t handle, float *dropout,
                                        void **states,
                                        unsigned long long *seed) {
  CUDNN_ENTER_API(cudnnGetDropoutDescriptor);
  cudnnStatus_t ret = func(dropoutDesc, handle, dropout, states, seed);
  return ret;
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,
                                  const cudnnDropoutDescriptor_t dropoutDesc,
                                  const cudnnTensorDescriptor_t xdesc,
                                  const void *x,
                                  const cudnnTensorDescriptor_t ydesc, void *y,
                                  void *reserveSpace,
                                  size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnDropoutForward);
  fromCudnnInputTensor(xdesc, x);
  fromCudnnOutputTensor(ydesc, y);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnDropoutForward, handle);
  cudnnStatus_t ret = func(handle, dropoutDesc, xdesc, x, ydesc, y,
                           reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,
                                   const cudnnDropoutDescriptor_t dropoutDesc,
                                   const cudnnTensorDescriptor_t dydesc,
                                   const void *dy,
                                   const cudnnTensorDescriptor_t dxdesc,
                                   void *dx, void *reserveSpace,
                                   size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnDropoutBackward);
  fromCudnnInputTensor(dydesc, dy);
  fromCudnnOutputTensor(dxdesc, dx);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnDropoutBackward, handle);
  cudnnStatus_t ret = func(handle, dropoutDesc, dydesc, dy, dxdesc, dx,
                           reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnCreateRNNDescriptor);
  cudnnStatus_t ret = func(rnnDesc);
  return ret;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnDestroyRNNDescriptor);
  cudnnStatus_t ret = func(rnnDesc);
  return ret;
}

cudnnStatus_t cudnnSetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  CUDNN_ENTER_API(cudnnSetRNNDescriptor);
  cudnnStatus_t ret = func(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                           inputMode, direction, mode, algo, mathPrec);
  return ret;
}

cudnnStatus_t cudnnGetRNNDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize,
    int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc,
    cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction,
    cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
  CUDNN_ENTER_API(cudnnGetRNNDescriptor);
  cudnnStatus_t ret = func(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                           inputMode, direction, mode, algo, mathPrec);
  return ret;
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                        cudnnMathType_t mType) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetRNNMatrixMathType);
  cudnnStatus_t ret = func(rnnDesc, mType);
  return ret;
}

cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                        cudnnMathType_t *mType) {
  CUDNN_ENTER_API(cudnnGetRNNMatrixMathType);
  cudnnStatus_t ret = func(rnnDesc, mType);
  return ret;
}

cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                  cudnnRNNBiasMode_t biasMode) {
  CUDNN_ENTER_API(cudnnSetRNNBiasMode);
  cudnnStatus_t ret = func(rnnDesc, biasMode);
  return ret;
}

cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                  cudnnRNNBiasMode_t *biasMode) {
  CUDNN_ENTER_API(cudnnGetRNNBiasMode);
  cudnnStatus_t ret = func(rnnDesc, biasMode);
  return ret;
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t handle,
                              cudnnRNNDescriptor_t rnnDesc,
                              cudnnRNNClipMode_t clipMode,
                              cudnnNanPropagation_t clipNanOpt, double lclip,
                              double rclip) {
  CUDNN_ENTER_API(cudnnRNNSetClip);
  cudnnStatus_t ret = func(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
  return ret;
}

cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t handle,
                              cudnnRNNDescriptor_t rnnDesc,
                              cudnnRNNClipMode_t *clipMode,
                              cudnnNanPropagation_t *clipNanOpt, double *lclip,
                              double *rclip) {
  CUDNN_ENTER_API(cudnnRNNGetClip);
  cudnnStatus_t ret = func(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
  return ret;
}

cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                          cudnnRNNDescriptor_t rnnDesc,
                                          const int recProjSize,
                                          const int outProjSize) {
  CUDNN_ENTER_API(cudnnSetRNNProjectionLayers);
  cudnnStatus_t ret = func(handle, rnnDesc, recProjSize, outProjSize);
  return ret;
}

cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                          const cudnnRNNDescriptor_t rnnDesc,
                                          int *recProjSize, int *outProjSize) {
  CUDNN_ENTER_API(cudnnGetRNNProjectionLayers);
  cudnnStatus_t ret = func(handle, rnnDesc, recProjSize, outProjSize);
  return ret;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                           const int minibatch,
                                           const cudnnDataType_t dataType,
                                           cudnnPersistentRNNPlan_t *plan) {
  CUDNN_ENTER_API(cudnnCreatePersistentRNNPlan);
  cudnnStatus_t ret = func(rnnDesc, minibatch, dataType, plan);
  return ret;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
  CUDNN_ENTER_API(cudnnDestroyPersistentRNNPlan);
  cudnnStatus_t ret = func(plan);
  return ret;
}

cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                        cudnnPersistentRNNPlan_t plan) {
  CUDNN_ENTER_API(cudnnSetPersistentRNNPlan);
  cudnnStatus_t ret = func(rnnDesc, plan);
  return ret;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t *xDesc,
                                       size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetRNNWorkspaceSize);
  cudnnStatus_t ret = func(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetRNNTrainingReserveSize);
  cudnnStatus_t ret = func(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnnDesc,
                                    const cudnnTensorDescriptor_t xDesc,
                                    size_t *sizeInBytes,
                                    cudnnDataType_t dataType) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnGetRNNParamsSize);
  cudnnStatus_t ret = func(handle, rnnDesc, xDesc, sizeInBytes, dataType);
  return ret;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
  CUDNN_ENTER_API(cudnnGetRNNLinLayerMatrixParams);
  fromCudnnInputFilter(wDesc, w);
  cudnnStatus_t ret = func(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w,
                           linLayerID, linLayerMatDesc, linLayerMat);
  return ret;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int pseudoLayer, const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
  CUDNN_ENTER_API(cudnnGetRNNLinLayerBiasParams);
  fromCudnnInputFilter(wDesc, w);
  cudnnStatus_t ret = func(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w,
                           linLayerID, linLayerBiasDesc, linLayerBias);
  return ret;
}

cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnRNNForwardInference);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNForwardInference, handle);
  cudnnStatus_t ret = func(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                           cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc,
                           cy, workspace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnRNNForwardTraining(cudnnHandle_t handle,
                        const cudnnRNNDescriptor_t rnnDesc, const int seqLength,
                        const cudnnTensorDescriptor_t *xDesc, const void *x,
                        const cudnnTensorDescriptor_t hxDesc, const void *hx,
                        const cudnnTensorDescriptor_t cxDesc, const void *cx,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnTensorDescriptor_t *yDesc, void *y,
                        const cudnnTensorDescriptor_t hyDesc, void *hy,
                        const cudnnTensorDescriptor_t cyDesc, void *cy,
                        void *workspace, size_t workSpaceSizeInBytes,
                        void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnRNNForwardTraining);
  fromCudnnInputTensorArray(xDesc, x, seqLength);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensorArray(yDesc, y, seqLength);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNForwardTraining, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc,
           w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workSpaceSizeInBytes,
           reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                     const int seqLength, const cudnnTensorDescriptor_t *yDesc,
                     const void *y, const cudnnTensorDescriptor_t *dyDesc,
                     const void *dy, const cudnnTensorDescriptor_t dhyDesc,
                     const void *dhy, const cudnnTensorDescriptor_t dcyDesc,
                     const void *dcy, const cudnnFilterDescriptor_t wDesc,
                     const void *w, const cudnnTensorDescriptor_t hxDesc,
                     const void *hx, const cudnnTensorDescriptor_t cxDesc,
                     const void *cx, const cudnnTensorDescriptor_t *dxDesc,
                     void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
                     const cudnnTensorDescriptor_t dcxDesc, void *dcx,
                     void *workspace, size_t workSpaceSizeInBytes,
                     void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnRNNBackwardData);
  fromCudnnInputTensorArray(yDesc, y, seqLength);
  fromCudnnInputTensorArray(dyDesc, dy, seqLength);
  fromCudnnInputTensor(dhyDesc, dhy);
  fromCudnnInputTensor(dcyDesc, dcy);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnOutputTensorArray(dxDesc, dx, seqLength);
  fromCudnnOutputTensor(dhxDesc, dhx);
  fromCudnnOutputTensor(dcxDesc, dcx);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNBackwardData, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy,
           dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc,
           dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes, reserveSpace,
           reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnRNNBackwardWeights(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnRNNBackwardWeights);
  fromCudnnInputTensorArray(xDesc, x, seqLength);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensorArray(yDesc, y, seqLength);
  fromCudnnOutputFilter(dwDesc, dw);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNBackwardWeights, handle);
  cudnnStatus_t ret = func(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx,
                           yDesc, y, workspace, workSpaceSizeInBytes, dwDesc,
                           dw, reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                     cudnnRNNPaddingMode_t paddingMode) {
  CUDNN_ENTER_API(cudnnSetRNNPaddingMode);
  cudnnStatus_t ret = func(rnnDesc, paddingMode);
  return ret;
}

cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                     cudnnRNNPaddingMode_t *paddingMode) {
  CUDNN_ENTER_API(cudnnGetRNNPaddingMode);
  cudnnStatus_t ret = func(rnnDesc, paddingMode);
  return ret;
}

cudnnStatus_t
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
  CUDNN_ENTER_API(cudnnCreateRNNDataDescriptor);
  cudnnStatus_t ret = func(rnnDataDesc);
  return ret;
}

cudnnStatus_t
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
  CUDNN_ENTER_API(cudnnDestroyRNNDataDescriptor);
  cudnnStatus_t ret = func(rnnDataDesc);
  return ret;
}

cudnnStatus_t
cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t dataType, cudnnRNNDataLayout_t layout,
                          int maxSeqLength, int batchSize, int vectorSize,
                          const int seqLengthArray[], void *paddingFill) {
  CUDNN_ENTER_API(cudnnSetRNNDataDescriptor);
  cudnnStatus_t ret = func(rnnDataDesc, dataType, layout, maxSeqLength,
                           batchSize, vectorSize, seqLengthArray, paddingFill);
  return ret;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType,
    cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize,
    int *vectorSize, int arrayLengthRequested, int seqLengthArray[],
    void *paddingFill) {
  CUDNN_ENTER_API(cudnnGetRNNDataDescriptor);
  cudnnStatus_t ret =
      func(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize,
           arrayLengthRequested, seqLengthArray, paddingFill);
  return ret;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnRNNForwardTrainingEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNForwardTrainingEx, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc,
           y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn,
           qDesc, queries, workSpace, workSpaceSizeInBytes, reserveSpace,
           reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys,
    const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void *queries, void *workSpace,
    size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnRNNForwardInferenceEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNForwardInferenceEx, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc,
           y, hyDesc, hy, cyDesc, cy, kDesc, keys, cDesc, cAttn, iDesc, iAttn,
           qDesc, queries, workSpace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnRNNBackwardDataEx(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
                       const cudnnRNNDataDescriptor_t yDesc, const void *y,
                       const cudnnRNNDataDescriptor_t dyDesc, const void *dy,
                       const cudnnRNNDataDescriptor_t dcDesc,
                       const void *dcAttn,
                       const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
                       const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
                       const cudnnFilterDescriptor_t wDesc, const void *w,
                       const cudnnTensorDescriptor_t hxDesc, const void *hx,
                       const cudnnTensorDescriptor_t cxDesc, const void *cx,
                       const cudnnRNNDataDescriptor_t dxDesc, void *dx,
                       const cudnnTensorDescriptor_t dhxDesc, void *dhx,
                       const cudnnTensorDescriptor_t dcxDesc, void *dcx,
                       const cudnnRNNDataDescriptor_t dkDesc, void *dkeys,
                       void *workSpace, size_t workSpaceSizeInBytes,
                       void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnRNNBackwardDataEx);
  fromCudnnInputTensor(dhyDesc, dhy);
  fromCudnnInputTensor(dcyDesc, dcy);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnOutputTensor(dhxDesc, dhx);
  fromCudnnOutputTensor(dcxDesc, dcx);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNBackwardDataEx, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy,
           dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc,
           dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace, workSpaceSizeInBytes,
           reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnRNNDataDescriptor_t yDesc, const void *y, void *workSpace,
    size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnRNNBackwardWeightsEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnWorkspace(workSpace, workSpaceSizeInBytes);
  fromCudnnOutputFilter(dwDesc, dw);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRNNBackwardWeightsEx, handle);
  cudnnStatus_t ret = func(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y,
                           workSpace, workSpaceSizeInBytes, dwDesc, dw,
                           reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle,
                               cudnnRNNDescriptor_t rnnDesc,
                               cudnnAlgorithmDescriptor_t algoDesc) {
  CUDNN_ENTER_API(cudnnSetRNNAlgorithmDescriptor);
  cudnnStatus_t ret = func(handle, rnnDesc, algoDesc);
  return ret;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  CUDNN_ENTER_API(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, rnnDesc, count);
  return ret;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnFindRNNForwardInferenceAlgorithmEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindRNNForwardInferenceAlgorithmEx, handle);
  cudnnStatus_t ret = func(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
      yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity, requestedAlgoCount,
      returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  CUDNN_ENTER_API(cudnnGetRNNForwardTrainingAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, rnnDesc, count);
  return ret;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy,
    const cudnnTensorDescriptor_t cyDesc, void *cy, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnFindRNNForwardTrainingAlgorithmEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnOutputTensor(hyDesc, hy);
  fromCudnnOutputTensor(cyDesc, cy);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindRNNForwardTrainingAlgorithmEx, handle);
  cudnnStatus_t ret =
      func(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc,
           w, yDesc, y, hyDesc, hy, cyDesc, cy, findIntensity,
           requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
           workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  CUDNN_ENTER_API(cudnnGetRNNBackwardDataAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, rnnDesc, count);
  return ret;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *yDesc, const void *y,
    const cudnnTensorDescriptor_t *dyDesc, const void *dy,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy,
    const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnTensorDescriptor_t *dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity,
    const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnFindRNNBackwardDataAlgorithmEx);
  fromCudnnInputTensor(dhyDesc, dhy);
  fromCudnnInputTensor(dcyDesc, dcy);
  fromCudnnInputFilter(wDesc, w);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnInputTensor(cxDesc, cx);
  fromCudnnOutputTensor(dhxDesc, dhx);
  fromCudnnOutputTensor(dcxDesc, dcx);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  fromCudnnWorkspace(reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindRNNBackwardDataAlgorithmEx, handle);
  cudnnStatus_t ret = func(
      handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc,
      dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc,
      dcx, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults,
      workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
  CUDNN_ENTER_API(cudnnGetRNNBackwardWeightsAlgorithmMaxCount);
  cudnnStatus_t ret = func(handle, rnnDesc, count);
  return ret;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t *yDesc, const void *y,
    const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults,
    const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnFindRNNBackwardWeightsAlgorithmEx);
  fromCudnnInputTensor(hxDesc, hx);
  fromCudnnOutputFilter(dwDesc, dw);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnFindRNNBackwardWeightsAlgorithmEx, handle);
  cudnnStatus_t ret = func(
      handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity,
      requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
      workSpaceSizeInBytes, dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) {
  CUDNN_ENTER_API(cudnnCreateSeqDataDescriptor);
  cudnnStatus_t ret = func(seqDataDesc);
  return ret;
}

cudnnStatus_t
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) {
  CUDNN_ENTER_API(cudnnDestroySeqDataDescriptor);
  cudnnStatus_t ret = func(seqDataDesc);
  return ret;
}

cudnnStatus_t cudnnSetSeqDataDescriptor(
    cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType, int nbDims,
    const int dimA[], const cudnnSeqDataAxis_t axes[],
    size_t seqLengthArraySize, const int seqLengthArray[], void *paddingFill) {
  CUDNN_ENTER_API(cudnnSetSeqDataDescriptor);
  cudnnStatus_t ret = func(seqDataDesc, dataType, nbDims, dimA, axes,
                           seqLengthArraySize, seqLengthArray, paddingFill);
  return ret;
}

cudnnStatus_t cudnnGetSeqDataDescriptor(
    const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType,
    int *nbDims, int nbDimsRequested, int dimA[], cudnnSeqDataAxis_t axes[],
    size_t *seqLengthArraySize, size_t seqLengthSizeRequested,
    int seqLengthArray[], void *paddingFill) {
  CUDNN_ENTER_API(cudnnGetSeqDataDescriptor);
  cudnnStatus_t ret = func(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA,
                           axes, seqLengthArraySize, seqLengthSizeRequested,
                           seqLengthArray, paddingFill);
  return ret;
}

cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
  CUDNN_ENTER_API(cudnnCreateAttnDescriptor);
  cudnnStatus_t ret = func(attnDesc);
  return ret;
}

cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
  CUDNN_ENTER_API(cudnnDestroyAttnDescriptor);
  cudnnStatus_t ret = func(attnDesc);
  return ret;
}

cudnnStatus_t cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads,
    double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec,
    cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize,
    int qProjSize, int kProjSize, int vProjSize, int oProjSize,
    int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize) {
  CUDNN_ENTER_API(cudnnSetAttnDescriptor);
  cudnnStatus_t ret =
      func(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec,
           mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize,
           qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
           kvMaxSeqLength, maxBatchSize, maxBeamSize);
  return ret;
}

cudnnStatus_t cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads,
    double *smScaler, cudnnDataType_t *dataType, cudnnDataType_t *computePrec,
    cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize,
    int *vSize, int *qProjSize, int *kProjSize, int *vProjSize, int *oProjSize,
    int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize,
    int *maxBeamSize) {
  CUDNN_ENTER_API(cudnnGetAttnDescriptor);
  cudnnStatus_t ret =
      func(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec,
           mathType, attnDropoutDesc, postDropoutDesc, qSize, kSize, vSize,
           qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
           kvMaxSeqLength, maxBatchSize, maxBeamSize);
  return ret;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                                           const cudnnAttnDescriptor_t attnDesc,
                                           size_t *weightSizeInBytes,
                                           size_t *workSpaceSizeInBytes,
                                           size_t *reserveSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnGetMultiHeadAttnBuffers);
  cudnnStatus_t ret = func(handle, attnDesc, weightSizeInBytes,
                           workSpaceSizeInBytes, reserveSpaceSizeInBytes);
  return ret;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes,
    const void *weights, cudnnTensorDescriptor_t wDesc, void **wAddr) {
  CUDNN_ENTER_API(cudnnGetMultiHeadAttnWeights);
  cudnnStatus_t ret =
      func(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr);
  return ret;
}

cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx,
    const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsQO[],
    const int devSeqLengthsKV[], const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const void *residuals,
    const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes,
    const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  CUDNN_ENTER_API(cudnnMultiHeadAttnForward);
  fromCudnnWorkspace(out, weightSizeInBytes);
  fromCudnnWorkspace(workSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnMultiHeadAttnForward, handle);
  cudnnStatus_t ret =
      func(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO,
           devSeqLengthsKV, qDesc, queries, residuals, kDesc, keys, vDesc,
           values, oDesc, out, weightSizeInBytes, weights, workSpaceSizeInBytes,
           workSpace, reserveSpaceSizeInBytes, reserveSpace);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    const int loWinIdx[], const int hiWinIdx[], const int devSeqLengthsDQDO[],
    const int devSeqLengthsDKDV[], const cudnnSeqDataDescriptor_t doDesc,
    const void *dout, const cudnnSeqDataDescriptor_t dqDesc, void *dqueries,
    const void *queries, const cudnnSeqDataDescriptor_t dkDesc, void *dkeys,
    const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues,
    const void *values, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  CUDNN_ENTER_API(cudnnMultiHeadAttnBackwardData);
  fromCudnnWorkspace(workSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnMultiHeadAttnBackwardData, handle);
  cudnnStatus_t ret = func(
      handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO,
      devSeqLengthsDKDV, doDesc, dout, dqDesc, dqueries, queries, dkDesc, dkeys,
      keys, dvDesc, dvalues, values, weightSizeInBytes, weights,
      workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc,
    cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc,
    const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values,
    const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights,
    size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
  CUDNN_ENTER_API(cudnnMultiHeadAttnBackwardWeights);
  fromCudnnWorkspace(dweights, workSpaceSizeInBytes);
  fromCudnnWorkspace(workSpace, reserveSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnMultiHeadAttnBackwardWeights, handle);
  cudnnStatus_t ret = func(handle, attnDesc, addGrad, qDesc, queries, kDesc,
                           keys, vDesc, values, doDesc, dout, weightSizeInBytes,
                           weights, dweights, workSpaceSizeInBytes, workSpace,
                           reserveSpaceSizeInBytes, reserveSpace);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
  CUDNN_ENTER_API(cudnnCreateCTCLossDescriptor);
  cudnnStatus_t ret = func(ctcLossDesc);
  return ret;
}

cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                        cudnnDataType_t compType) {
  CUDNN_ENTER_API(cudnnSetCTCLossDescriptor);
  cudnnStatus_t ret = func(ctcLossDesc, compType);
  return ret;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                                          cudnnDataType_t compType,
                                          cudnnLossNormalizationMode_t normMode,
                                          cudnnNanPropagation_t gradMode) {
  CUDNN_ENTER_API(cudnnSetCTCLossDescriptorEx);
  cudnnStatus_t ret = func(ctcLossDesc, compType, normMode, gradMode);
  return ret;
}

cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                        cudnnDataType_t *compType) {
  CUDNN_ENTER_API(cudnnGetCTCLossDescriptor);
  cudnnStatus_t ret = func(ctcLossDesc, compType);
  return ret;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(
    cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType,
    cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode) {
  CUDNN_ENTER_API(cudnnGetCTCLossDescriptorEx);
  cudnnStatus_t ret = func(ctcLossDesc, compType, normMode, gradMode);
  return ret;
}

cudnnStatus_t
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
  CUDNN_ENTER_API(cudnnDestroyCTCLossDescriptor);
  cudnnStatus_t ret = func(ctcLossDesc);
  return ret;
}

cudnnStatus_t
cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
             const void *probs, const int *labels, const int *labelLengths,
             const int *inputLengths, void *costs,
             const cudnnTensorDescriptor_t gradientsDesc, const void *gradients,
             cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc,
             void *workspace, size_t workSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnCTCLoss);
  fromCudnnInputTensor(probsDesc, probs);
  fromCudnnInputTensor(gradientsDesc, gradients);
  fromCudnnWorkspace(workspace, workSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnCTCLoss, handle);
  cudnnStatus_t ret = func(handle, probsDesc, probs, labels, labelLengths,
                           inputLengths, costs, gradientsDesc, gradients, algo,
                           ctcLossDesc, workspace, workSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
    const cudnnTensorDescriptor_t gradientsDesc, const int *labels,
    const int *labelLengths, const int *inputLengths, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
  CUDNN_ENTER_API(cudnnGetCTCLossWorkspaceSize);
  cudnnStatus_t ret =
      func(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths,
           algo, ctcLossDesc, sizeInBytes);
  return ret;
}

cudnnStatus_t
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
  CUDNN_ENTER_API(cudnnCreateAlgorithmDescriptor);
  cudnnStatus_t ret = func(algoDesc);
  return ret;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc,
                                          cudnnAlgorithm_t algorithm) {
  CUDNN_ENTER_API(cudnnSetAlgorithmDescriptor);
  cudnnStatus_t ret = func(algoDesc, algorithm);
  return ret;
}

cudnnStatus_t
cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc,
                            cudnnAlgorithm_t *algorithm) {
  CUDNN_ENTER_API(cudnnGetAlgorithmDescriptor);
  cudnnStatus_t ret = func(algoDesc, algorithm);
  return ret;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src,
                                           cudnnAlgorithmDescriptor_t dest) {
  CUDNN_ENTER_API(cudnnCopyAlgorithmDescriptor);
  cudnnStatus_t ret = func(src, dest);
  return ret;
}

cudnnStatus_t
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
  CUDNN_ENTER_API(cudnnDestroyAlgorithmDescriptor);
  cudnnStatus_t ret = func(algoDesc);
  return ret;
}

cudnnStatus_t
cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf,
                                int numberToCreate) {
  CUDNN_ENTER_API(cudnnCreateAlgorithmPerformance);
  cudnnStatus_t ret = func(algoPerf, numberToCreate);
  return ret;
}

cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                           cudnnAlgorithmDescriptor_t algoDesc,
                                           cudnnStatus_t status, float time,
                                           size_t memory) {
  CUDNN_ENTER_API(cudnnSetAlgorithmPerformance);
  cudnnStatus_t ret = func(algoPerf, algoDesc, status, time, memory);
  return ret;
}

cudnnStatus_t
cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t *algoDesc,
                             cudnnStatus_t *status, float *time,
                             size_t *memory) {
  CUDNN_ENTER_API(cudnnGetAlgorithmPerformance);
  cudnnStatus_t ret = func(algoPerf, algoDesc, status, time, memory);
  return ret;
}

cudnnStatus_t
cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf,
                                 int numberToDestroy) {
  CUDNN_ENTER_API(cudnnDestroyAlgorithmPerformance);
  cudnnStatus_t ret = func(algoPerf, numberToDestroy);
  return ret;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle,
                                         cudnnAlgorithmDescriptor_t algoDesc,
                                         size_t *algoSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnGetAlgorithmSpaceSize);
  cudnnStatus_t ret = func(handle, algoDesc, algoSpaceSizeInBytes);
  return ret;
}

cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t handle,
                                 cudnnAlgorithmDescriptor_t algoDesc,
                                 void *algoSpace, size_t algoSpaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnSaveAlgorithm);
  fromCudnnWorkspace(algoSpace, algoSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnSaveAlgorithm, handle);
  cudnnStatus_t ret = func(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t handle, void *algoSpace,
                                    size_t algoSpaceSizeInBytes,
                                    cudnnAlgorithmDescriptor_t algoDesc) {
  CUDNN_ENTER_API(cudnnRestoreAlgorithm);
  fromCudnnWorkspace(algoSpace, algoSpaceSizeInBytes);
  CUDNN_ENTER_COMPUTE_KERNEL(cudnnRestoreAlgorithm, handle);
  cudnnStatus_t ret = func(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
  CUDNN_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cudnnStatus_t cudnnSetCallback(unsigned mask, void *udata,
                               cudnnCallback_t fptr) {
  CUDNN_ENTER_API(cudnnSetCallback);
  cudnnStatus_t ret = func(mask, udata, fptr);
  return ret;
}

cudnnStatus_t cudnnGetCallback(unsigned *mask, void **udata,
                               cudnnCallback_t *fptr) {
  CUDNN_ENTER_API(cudnnGetCallback);
  cudnnStatus_t ret = func(mask, udata, fptr);
  return ret;
}

cudnnStatus_t
cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack,
                                  cudnnFusedOps_t ops) {
  CUDNN_ENTER_API(cudnnCreateFusedOpsConstParamPack);
  cudnnStatus_t ret = func(constPack, ops);
  return ret;
}

cudnnStatus_t
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
  CUDNN_ENTER_API(cudnnDestroyFusedOpsConstParamPack);
  cudnnStatus_t ret = func(constPack);
  return ret;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(
    cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, const void *param) {
  CUDNN_ENTER_API(cudnnSetFusedOpsConstParamPackAttribute);
  cudnnStatus_t ret = func(constPack, paramLabel, param);
  return ret;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(
    const cudnnFusedOpsConstParamPack_t constPack,
    cudnnFusedOpsConstParamLabel_t paramLabel, void *param, int *isNULL) {
  CUDNN_ENTER_API(cudnnGetFusedOpsConstParamPackAttribute);
  cudnnStatus_t ret = func(constPack, paramLabel, param, isNULL);
  return ret;
}

cudnnStatus_t
cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack,
                                    cudnnFusedOps_t ops) {
  CUDNN_ENTER_API(cudnnCreateFusedOpsVariantParamPack);
  cudnnStatus_t ret = func(varPack, ops);
  return ret;
}

cudnnStatus_t
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
  CUDNN_ENTER_API(cudnnDestroyFusedOpsVariantParamPack);
  cudnnStatus_t ret = func(varPack);
  return ret;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(
    cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
  CUDNN_ENTER_API(cudnnSetFusedOpsVariantParamPackAttribute);
  cudnnStatus_t ret = func(varPack, paramLabel, ptr);
  return ret;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(
    const cudnnFusedOpsVariantParamPack_t varPack,
    cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
  CUDNN_ENTER_API(cudnnGetFusedOpsVariantParamPackAttribute);
  cudnnStatus_t ret = func(varPack, paramLabel, ptr);
  return ret;
}

cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan,
                                      cudnnFusedOps_t ops) {
  CUDNN_ENTER_API(cudnnCreateFusedOpsPlan);
  cudnnStatus_t ret = func(plan, ops);
  return ret;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
  CUDNN_ENTER_API(cudnnDestroyFusedOpsPlan);
  cudnnStatus_t ret = func(plan);
  return ret;
}

cudnnStatus_t
cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes) {
  CUDNN_ENTER_API(cudnnMakeFusedOpsPlan);
  cudnnStatus_t ret = func(handle, plan, constPack, workspaceSizeInBytes);
  return ret;
}

cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t handle,
                                   const cudnnFusedOpsPlan_t plan,
                                   cudnnFusedOpsVariantParamPack_t varPack) {
  CUDNN_ENTER_API(cudnnFusedOpsExecute);
  cudnnStatus_t ret = func(handle, plan, varPack);
  return ret;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(
    cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
    cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction,
    cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
  CUDNN_ENTER_API_OPTIMIZED(cudnnSetRNNDescriptor_v6);
  cudnnStatus_t ret = func(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc,
                           inputMode, direction, mode, algo, mathPrec);
  return ret;
}

cudnnStatus_t cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                                       int hiddenSize, int numLayers,
                                       cudnnDropoutDescriptor_t dropoutDesc,
                                       cudnnRNNInputMode_t inputMode,
                                       cudnnDirectionMode_t direction,
                                       cudnnRNNMode_t mode,
                                       cudnnDataType_t mathPrec) {
  CUDNN_ENTER_API(cudnnSetRNNDescriptor_v5);
  cudnnStatus_t ret = func(rnnDesc, hiddenSize, numLayers, dropoutDesc,
                           inputMode, direction, mode, mathPrec);
  return ret;
}