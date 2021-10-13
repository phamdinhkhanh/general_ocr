# Copyright (c) GeneralOCR. All rights reserved.
from general_ocr.runner import BaseModule

from general_ocr.models.builder import ENCODERS


@ENCODERS.register_module()
class BaseEncoder(BaseModule):
    """Base Encoder class for text recognition."""

    def forward(self, feat, **kwargs):
        return feat
