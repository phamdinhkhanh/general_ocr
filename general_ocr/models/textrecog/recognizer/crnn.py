# Copyright (c) GeneralOCR. All rights reserved.
from general_ocr.models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class CRNNNet(EncodeDecodeRecognizer):
    """CTC-loss based recognizer."""
