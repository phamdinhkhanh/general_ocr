# Copyright (c) GeneralOCR. All rights reserved.
from general_ocr.models.detectors import MaskRCNN

from general_ocr.models.builder import DETECTORS
from general_ocr.models.textdet.detectors.text_detector_mixin import \
    TextDetectorMixin


@DETECTORS.register_module()
class OCRMaskRCNN(TextDetectorMixin, MaskRCNN):
    """Mask RCNN tailored for OCR."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 text_repr_type='quad',
                 show_score=False,
                 init_cfg=None):
        TextDetectorMixin.__init__(self, show_score)
        MaskRCNN.__init__(
            self,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        assert text_repr_type in ['quad', 'poly']
        self.text_repr_type = text_repr_type

    def simple_test(self, img, img_metas, proposals=None, rescale=False):

        results = super().simple_test(img, img_metas, proposals, rescale)

        boundaries = self.get_boundary(results[0])
        boundaries = boundaries if isinstance(boundaries,
                                              list) else [boundaries]
        return boundaries
