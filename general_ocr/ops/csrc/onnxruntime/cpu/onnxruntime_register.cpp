// Copyright (c) OpenMMLab. All rights reserved
#include "onnxruntime_register.h"

#include "corner_pool.h"
#include "grid_sample.h"
#include "modulated_deform_conv.h"
#include "nms.h"
#include "ort_general_ocr_utils.h"
#include "reduce_ops.h"
#include "roi_align.h"
#include "roi_align_rotated.h"
#include "soft_nms.h"

const char *c_general_ocrOpDomain = "general_ocr";
SoftNmsOp c_SoftNmsOp;
NmsOp c_NmsOp;
general_ocrRoiAlignCustomOp c_general_ocrRoiAlignCustomOp;
general_ocrRoIAlignRotatedCustomOp c_general_ocrRoIAlignRotatedCustomOp;
GridSampleOp c_GridSampleOp;
general_ocrCumMaxCustomOp c_general_ocrCumMaxCustomOp;
general_ocrCumMinCustomOp c_general_ocrCumMinCustomOp;
general_ocrCornerPoolCustomOp c_general_ocrCornerPoolCustomOp;
general_ocrModulatedDeformConvOp c_general_ocrModulatedDeformConvOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_general_ocrOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_NmsOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_general_ocrRoiAlignCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_general_ocrRoIAlignRotatedCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_general_ocrCornerPoolCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_general_ocrCumMaxCustomOp)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_general_ocrCumMinCustomOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_general_ocrModulatedDeformConvOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
