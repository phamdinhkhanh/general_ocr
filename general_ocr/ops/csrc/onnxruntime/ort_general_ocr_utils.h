// Copyright (c) GeneralOCR. All rights reserved
#ifndef ORT_general_ocr_UTILS_H
#define ORT_general_ocr_UTILS_H
#include <onnxruntime_cxx_api.h>

#include <vector>

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};
#endif  // ORT_general_ocr_UTILS_H
