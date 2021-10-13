# Copyright (c) GeneralOCR. All rights reserved.
import general_ocr
import numpy as np
from general_ocr.datasets.builder import PIPELINES
from general_ocr.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class MultiRotateAugOCR:
    """Test-time augmentation with multiple rotations in the case that
    img_height > img_width.

    An example configuration is as follows:

    .. code-block::

        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=160,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ]

    After MultiRotateAugOCR with above configuration, the results are wrapped
    into lists of the same length as follows:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...]
            ...
        )

    Args:
        transforms (list[dict]): Transformation applied for each augmentation.
        rotate_degrees (list[int] | None): Degrees of anti-clockwise rotation.
        force_rotate (bool): If True, rotate image by 'rotate_degrees'
            while ignore image aspect ratio.
    """

    def __init__(self, transforms, rotate_degrees=None, force_rotate=False):
        self.transforms = Compose(transforms)
        self.force_rotate = force_rotate
        if rotate_degrees is not None:
            self.rotate_degrees = rotate_degrees if isinstance(
                rotate_degrees, list) else [rotate_degrees]
            assert general_ocr.is_list_of(self.rotate_degrees, int)
            for degree in self.rotate_degrees:
                assert 0 <= degree < 360
                assert degree % 90 == 0
            if 0 not in self.rotate_degrees:
                self.rotate_degrees.append(0)
        else:
            self.rotate_degrees = [0]

    def __call__(self, results):
        """Call function to apply test time augment transformation to results.

        Args:
            results (dict): Result dict contains the data to be transformed.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        img_shape = results['img_shape']
        ori_height, ori_width = img_shape[:2]
        if not self.force_rotate and ori_height <= ori_width:
            rotate_degrees = [0]
        else:
            rotate_degrees = self.rotate_degrees
        aug_data = []
        for degree in set(rotate_degrees):
            _results = results.copy()
            if degree == 0:
                pass
            elif degree == 90:
                _results['img'] = np.rot90(_results['img'], 1)
            elif degree == 180:
                _results['img'] = np.rot90(_results['img'], 2)
            elif degree == 270:
                _results['img'] = np.rot90(_results['img'], 3)
            data = self.transforms(_results)
            aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'rotate_degrees={self.rotate_degrees})'
        return repr_str



@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.
    An example configuration is as followed:
    .. code-block::
        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert general_ocr.is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert general_ocr.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and general_ocr.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert general_ocr.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert general_ocr.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.
        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        if self.img_scale is None and general_ocr.is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str