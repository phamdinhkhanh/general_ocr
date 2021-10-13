# Copyright (c) GeneralOCR. All rights reserved.
from . import utils
from .base_dataset import BaseDataset
from .icdar_dataset import IcdarDataset
from .kie_dataset import KIEDataset
from .ner_dataset import NerDataset
from .ocr_dataset import OCRDataset
from .ocr_seg_dataset import OCRSegDataset
from .pipelines import CustomFormatBundle, DBNetTargets, FCENetTargets
from .text_det_dataset import TextDetDataset
from .uniform_concat_dataset import UniformConcatDataset

from .utils import *  # NOQA

# ---------------
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset
from .custom import CustomDataset
from .dataset_wrappers import (ClassBalancedDataset, ConcatDataset,
                               MultiImageMixDataset, RepeatDataset)
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from .utils_ import (NumClassCheckHook, get_loading_pipeline,
                    replace_ImageToTensor)
from .voc import VOCDataset
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

__all__ = [
    'DATASETS', 'IcdarDataset', 'build_dataloader', 'build_dataset',
    'BaseDataset', 'OCRDataset', 'TextDetDataset', 'CustomFormatBundle',
    'DBNetTargets', 'OCRSegDataset', 'KIEDataset', 'FCENetTargets',
    'NerDataset', 'UniformConcatDataset', 'CustomDataset', 'XMLDataset', 'CocoDataset', 'DeepFashionDataset',
    'VOCDataset', 'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset',
    'LVISV1Dataset', 'GroupSampler', 'DistributedGroupSampler',
    'DistributedSampler', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'ClassBalancedDataset', 'WIDERFaceDataset', 'DATASETS', 'PIPELINES',
    'build_dataset', 'replace_ImageToTensor', 'get_loading_pipeline',
    'NumClassCheckHook', 'CocoPanopticDataset', 'MultiImageMixDataset'
]

__all__ += utils.__all__
