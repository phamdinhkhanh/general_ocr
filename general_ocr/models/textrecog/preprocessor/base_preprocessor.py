# Copyright (c) GeneralOCR. All rights reserved.
from general_ocr.runner import BaseModule

from general_ocr.models.builder import PREPROCESSOR


@PREPROCESSOR.register_module()
class BasePreprocessor(BaseModule):
    """Base Preprocessor class for text recognition."""

    def forward(self, x, **kwargs):
        return x
