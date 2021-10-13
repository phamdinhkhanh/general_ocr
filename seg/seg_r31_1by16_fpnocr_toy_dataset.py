checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
gt_label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomPaddingOCR',
        max_ratio=[0.15, 0.2, 0.15, 0.2],
        box_type='char_quads'),
    dict(type='OpencvToPil'),
    dict(
        type='RandomRotateImageBox',
        min_angle=-17,
        max_angle=17,
        box_type='char_quads'),
    dict(type='PilToOpencv'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=512,
        keep_aspect_ratio=True),
    dict(
        type='OCRSegTargets',
        label_convertor=dict(
            type='SegConvertor',
            dict_type='DICT36',
            with_unknown=True,
            lower=True),
        box_type='char_quads'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='ToTensorOCR'),
    dict(type='FancyPCA'),
    dict(
        type='NormalizeOCR',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels'],
        visualize=dict(flag=False, boundary_key=None),
        call_super=False),
    dict(
        type='Collect',
        keys=['img', 'gt_kernels'],
        meta_keys=['filename', 'ori_shape', 'img_shape'])
]
test_img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.120000000000005, 57.375])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=None,
        keep_aspect_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.120000000000005, 57.375]),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'resize_shape'])
]
prefix = 'tests/data/ocr_char_ann_toy_dataset/'
train = dict(
    type='OCRSegDataset',
    img_prefix='tests/data/ocr_char_ann_toy_dataset/imgs',
    ann_file='tests/data/ocr_char_ann_toy_dataset/instances_train.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=100,
        parser=dict(
            type='LineJsonParser', keys=['file_name', 'annotations', 'text'])),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='RandomPaddingOCR',
            max_ratio=[0.15, 0.2, 0.15, 0.2],
            box_type='char_quads'),
        dict(type='OpencvToPil'),
        dict(
            type='RandomRotateImageBox',
            min_angle=-17,
            max_angle=17,
            box_type='char_quads'),
        dict(type='PilToOpencv'),
        dict(
            type='ResizeOCR',
            height=64,
            min_width=64,
            max_width=512,
            keep_aspect_ratio=True),
        dict(
            type='OCRSegTargets',
            label_convertor=dict(
                type='SegConvertor',
                dict_type='DICT36',
                with_unknown=True,
                lower=True),
            box_type='char_quads'),
        dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
        dict(type='ToTensorOCR'),
        dict(type='FancyPCA'),
        dict(
            type='NormalizeOCR',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        dict(
            type='CustomFormatBundle',
            keys=['gt_kernels'],
            visualize=dict(flag=False, boundary_key=None),
            call_super=False),
        dict(
            type='Collect',
            keys=['img', 'gt_kernels'],
            meta_keys=['filename', 'ori_shape', 'img_shape'])
    ],
    test_mode=True)
test = dict(
    type='OCRDataset',
    img_prefix='tests/data/ocr_char_ann_toy_dataset/imgs',
    ann_file='tests/data/ocr_char_ann_toy_dataset/instances_test.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='ResizeOCR',
            height=64,
            min_width=64,
            max_width=None,
            keep_aspect_ratio=True),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.120000000000005, 57.375]),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=['filename', 'ori_shape', 'resize_shape'])
    ],
    test_mode=True)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type='OCRSegDataset',
        img_prefix='tests/data/ocr_char_ann_toy_dataset/imgs',
        ann_file='tests/data/ocr_char_ann_toy_dataset/instances_train.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=100,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'annotations', 'text'])),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomPaddingOCR',
                max_ratio=[0.15, 0.2, 0.15, 0.2],
                box_type='char_quads'),
            dict(type='OpencvToPil'),
            dict(
                type='RandomRotateImageBox',
                min_angle=-17,
                max_angle=17,
                box_type='char_quads'),
            dict(type='PilToOpencv'),
            dict(
                type='ResizeOCR',
                height=64,
                min_width=64,
                max_width=512,
                keep_aspect_ratio=True),
            dict(
                type='OCRSegTargets',
                label_convertor=dict(
                    type='SegConvertor',
                    dict_type='DICT36',
                    with_unknown=True,
                    lower=True),
                box_type='char_quads'),
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            dict(type='ToTensorOCR'),
            dict(type='FancyPCA'),
            dict(
                type='NormalizeOCR',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='CustomFormatBundle',
                keys=['gt_kernels'],
                visualize=dict(flag=False, boundary_key=None),
                call_super=False),
            dict(
                type='Collect',
                keys=['img', 'gt_kernels'],
                meta_keys=['filename', 'ori_shape', 'img_shape'])
        ],
        test_mode=True),
    val=dict(
        type='OCRDataset',
        img_prefix='tests/data/ocr_char_ann_toy_dataset/imgs',
        ann_file='tests/data/ocr_char_ann_toy_dataset/instances_test.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=64,
                min_width=64,
                max_width=None,
                keep_aspect_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.120000000000005, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=['filename', 'ori_shape', 'resize_shape'])
        ],
        test_mode=True),
    test=dict(
        type='OCRDataset',
        img_prefix='tests/data/ocr_char_ann_toy_dataset/imgs',
        ann_file='tests/data/ocr_char_ann_toy_dataset/instances_test.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='LineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator=' ')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=64,
                min_width=64,
                max_width=None,
                keep_aspect_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.120000000000005, 57.375]),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=['filename', 'ori_shape', 'resize_shape'])
        ],
        test_mode=True))
evaluation = dict(interval=1, metric='acc')
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5
label_convertor = dict(
    type='SegConvertor', dict_type='DICT36', with_unknown=True, lower=True)
model = dict(
    type='SegRecognizer',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        out_indices=[0, 1, 2, 3],
        stage4_pool_cfg=dict(kernel_size=2, stride=2),
        last_stage_pool=True),
    neck=dict(
        type='FPNOCR', in_channels=[128, 256, 512, 512], out_channels=256),
    head=dict(
        type='SegHead',
        in_channels=256,
        upsample_param=dict(scale_factor=2.0, mode='nearest')),
    loss=dict(
        type='SegLoss', seg_downsample_ratio=1.0, seg_with_loss_weight=False),
    label_convertor=dict(
        type='SegConvertor', dict_type='DICT36', with_unknown=True,
        lower=True))
find_unused_parameters = True
work_dir = 'seg'
gpu_ids = range(0, 1)
