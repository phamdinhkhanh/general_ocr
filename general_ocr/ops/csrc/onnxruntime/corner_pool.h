// Copyright (c) GeneralOCR. All rights reserved
#ifndef ONNXRUNTIME_CORNER_POOL_H
#define ONNXRUNTIME_CORNER_POOL_H

#include <assert.h>
#include <onnxruntime_cxx_api.h>

struct general_ocrCornerPoolKernel {
 public:
  general_ocrCornerPoolKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    mode_ = ort_.KernelInfoGetAttribute<int64_t>(info, "mode");
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;

  int64_t mode_;
};

struct general_ocrCornerPoolCustomOp
    : Ort::CustomOpBase<general_ocrCornerPoolCustomOp, general_ocrCornerPoolKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new general_ocrCornerPoolKernel(api, info);
  }

  const char* GetName() const { return "general_ocrCornerPool"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
};
#endif  // ONNXRUNTIME_CORNER_POOL_H
