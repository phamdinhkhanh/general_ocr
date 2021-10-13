// Copyright (c) OpenMMLab. All rights reserved
#ifndef ONNXRUNTIME_MODULATED_DEFORM_CONV_H
#define ONNXRUNTIME_MODULATED_DEFORM_CONV_H

#include <onnxruntime_cxx_api.h>

struct general_ocrModulatedDeformConvKernel {
  general_ocrModulatedDeformConvKernel(OrtApi api, const OrtKernelInfo *info);

  void Compute(OrtKernelContext *context);

 protected:
  OrtApi api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo *info_;
  Ort::AllocatorWithDefaultOptions allocator_;

  int64_t stride_height_;
  int64_t stride_width_;
  int64_t padding_height_;
  int64_t padding_width_;
  int64_t dilation_height_;
  int64_t dilation_width_;
  int64_t deformable_group_;
  int64_t group_;
};

struct general_ocrModulatedDeformConvOp
    : Ort::CustomOpBase<general_ocrModulatedDeformConvOp,
                        general_ocrModulatedDeformConvKernel> {
  void *CreateKernel(OrtApi api, const OrtKernelInfo *info) const {
    return new general_ocrModulatedDeformConvKernel(api, info);
  }

  const char *GetName() const { return "general_ocrModulatedDeformConv2d"; };

  size_t GetInputTypeCount() const { return 5; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(
      size_t index) const {
    // The last input (index == 4) is optional, which is bias
    if (index == 4)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // force cpu
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  };
};
#endif
