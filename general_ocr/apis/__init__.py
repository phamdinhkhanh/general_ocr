# Copyright (c) GeneralOCR. All rights reserved.
from .inference import init_detector, model_inference, inference_detector
from .train import train_detector

__all__ = ['model_inference', 'train_detector', 'init_detector', 'inference_detector']
