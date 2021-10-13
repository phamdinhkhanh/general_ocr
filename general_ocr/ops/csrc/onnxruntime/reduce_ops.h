// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_REDUCE_OPS_H
#define ONNXRUNTIME_REDUCE_OPS_H

#include <onnxruntime_cxx_api.h>

struct general_ocrCumMaxKernel {
 public:
  general_ocrCumMaxKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    dim_ = ort_.KernelInfoGetAttribute<int64_t>(info, "dim");

    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t dim_;
};

struct general_ocrCumMinKernel {
 public:
  general_ocrCumMinKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info)
      : ort_(ort) {
    dim_ = ort_.KernelInfoGetAttribute<int64_t>(info, "dim");

    // create allocator
    allocator_ = Ort::AllocatorWithDefaultOptions();
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t dim_;
};

struct general_ocrCumMaxCustomOp
    : Ort::CustomOpBase<general_ocrCumMaxCustomOp, general_ocrCumMaxKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new general_ocrCumMaxKernel(api, info);
  }

  const char* GetName() const { return "cummax"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

struct general_ocrCumMinCustomOp
    : Ort::CustomOpBase<general_ocrCumMinCustomOp, general_ocrCumMinKernel> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new general_ocrCumMinKernel(api, info);
  }

  const char* GetName() const { return "cummin"; }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    if (index == 1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char* GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};

#endif  // ONNXRUNTIME_REDUCE_OPS_H
