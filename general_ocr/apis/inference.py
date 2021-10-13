# Copyright (c) GeneralOCR. All rights reserved.
import warnings

import general_ocr
import numpy as np
import torch
# from general_ocr.ops import RoIPool
from general_ocr.parallel import collate, scatter
from general_ocr.runner import load_checkpoint
from general_ocr.core import get_classes
from general_ocr.datasets import replace_ImageToTensor
from general_ocr.datasets.pipelines import Compose
from general_ocr.ops import RoIPool
from general_ocr.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`general_ocr.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = general_ocr.Config.fromfile(config)
    elif not isinstance(config, general_ocr.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if config.model.get('pretrained'):
        config.model.pretrained = None
    config.model.train_cfg = None
    
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def disable_text_recog_aug_test(cfg, set_types=None):
    """Remove aug_test from test pipeline of text recognition.
    Args:
        cfg (general_ocr.Config): Input config.
        set_types (list[str]): Type of dataset source. Should be
            None or sublist of ['test', 'val']

    Returns:
        cfg (general_ocr.Config): Output config removing
            `MultiRotateAugOCR` in test pipeline.
    """
    assert set_types is None or isinstance(set_types, list)
    if set_types is None:
        set_types = ['val', 'test']
    for set_type in set_types:
        if cfg.data[set_type].pipeline[1].type == 'MultiRotateAugOCR':
            cfg.data[set_type].pipeline = [
                cfg.data[set_type].pipeline[0],
                *cfg.data[set_type].pipeline[1].transforms
            ]

    return cfg


def model_inference(model,
                    imgs,
                    ann=None,
                    batch_mode=False,
                    return_data=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
        ann (dict): Annotation info for key information extraction.
        return_data: Return postprocessed data.
    Returns:
        result (dict): Predicted results.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    is_ndarray = isinstance(imgs[0], np.ndarray)

    cfg = model.cfg

    if batch_mode:
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    device = next(model.parameters()).device  # model device

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(img=img, ann_info=ann, bbox_fields=[])
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img),
                img_prefix=None,
                ann_info=ann,
                bbox_fields=[])

        # build the data pipeline
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    # for KIE models
    if ann is not None:
        data['relations'] = data['relations'].data[0]
        data['gt_bboxes'] = data['gt_bboxes'].data[0]
        data['texts'] = data['texts'].data[0]
        data['img'] = data['img'][0]
        data['img_metas'] = data['img_metas'][0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        if not return_data:
            return results[0]
        return results[0], datas[0]
    else:
        if not return_data:
            return results
        return results, datas


def text_model_inference(model, input_sentence):
    """Inference text(s) with the entity recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        input_sentence (str): A text entered by the user.

    Returns:
        result (dict): Predicted results.
    """

    assert isinstance(input_sentence, str)

    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'text': input_sentence, 'label': {}}

    # build the data pipeline
    data = test_pipeline(data)
    if isinstance(data['img_metas'], dict):
        img_metas = data['img_metas']
    else:
        img_metas = data['img_metas'].data

    assert isinstance(img_metas, dict)
    img_metas = {
        'input_ids': img_metas['input_ids'].unsqueeze(0),
        'attention_masks': img_metas['attention_masks'].unsqueeze(0),
        'token_type_ids': img_metas['token_type_ids'].unsqueeze(0),
        'labels': img_metas['labels'].unsqueeze(0)
    }
    # forward the model
    with torch.no_grad():
        result = model(None, img_metas, return_loss=False)
    return result


def inference_detector(model, imgs):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.
    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results