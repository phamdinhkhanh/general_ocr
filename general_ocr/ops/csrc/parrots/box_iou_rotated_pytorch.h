// Copyright (c) GeneralOCR. All rights reserved
#ifndef BOX_IOU_ROTATED_PYTORCH_H
#define BOX_IOU_ROTATED_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);

#ifdef GENERAL_OCR_WITH_CUDA
void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
#endif

#endif  // BOX_IOU_ROTATED_PYTORCH_H
