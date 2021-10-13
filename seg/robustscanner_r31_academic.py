checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)
hybrid_decoder = dict(type='SequenceAttentionDecoder')
position_decoder = dict(type='PositionAttentionDecoder')
model = dict(
    type='RobustScanner',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='ChannelReductionEncoder', in_channels=512, out_channels=128),
    decoder=dict(
        type='RobustScannerDecoder',
        dim_input=512,
        dim_model=128,
        hybrid_decoder=dict(type='SequenceAttentionDecoder'),
        position_decoder=dict(type='PositionAttentionDecoder')),
    loss=dict(type='SARLoss'),
    label_convertor=dict(
        type='AttnConvertor', dict_type='DICT90', with_unknown=True),
    max_seq_len=30)
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ])
        ])
]
dataset_type = 'OCRDataset'
train_prefix = 'data/mixture/'
train_img_prefix1 = 'data/mixture/icdar_2011'
train_img_prefix2 = 'data/mixture/icdar_2013'
train_img_prefix3 = 'data/mixture/icdar_2015'
train_img_prefix4 = 'data/mixture/coco_text'
train_img_prefix5 = 'data/mixture/III5K'
train_img_prefix6 = 'data/mixture/SynthText_Add'
train_img_prefix7 = 'data/mixture/SynthText'
train_img_prefix8 = 'data/mixture/Syn90k'
train_ann_file1 = ('data/mixture/icdar_2011/train_label.txt', )
train_ann_file2 = ('data/mixture/icdar_2013/train_label.txt', )
train_ann_file3 = ('data/mixture/icdar_2015/train_label.txt', )
train_ann_file4 = ('data/mixture/coco_text/train_label.txt', )
train_ann_file5 = ('data/mixture/III5K/train_label.txt', )
train_ann_file6 = ('data/mixture/SynthText_Add/label.txt', )
train_ann_file7 = ('data/mixture/SynthText/shuffle_labels.txt', )
train_ann_file8 = 'data/mixture/Syn90k/shuffle_labels.txt'
train1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2011',
    ann_file=('data/mixture/icdar_2011/train_label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2013',
    ann_file=('data/mixture/icdar_2013/train_label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train3 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2015',
    ann_file=('data/mixture/icdar_2015/train_label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train4 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/coco_text',
    ann_file=('data/mixture/coco_text/train_label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train5 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/III5K',
    ann_file=('data/mixture/III5K/train_label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train6 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/SynthText_Add',
    ann_file=('data/mixture/SynthText_Add/label.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train7 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/SynthText',
    ann_file=('data/mixture/SynthText/shuffle_labels.txt', ),
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
train8 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/Syn90k',
    ann_file='data/mixture/Syn90k/shuffle_labels.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
test_prefix = 'data/mixture/'
test_img_prefix1 = 'data/mixture/IIIT5K/'
test_img_prefix2 = 'data/mixture/svt/'
test_img_prefix3 = 'data/mixture/icdar_2013/'
test_img_prefix4 = 'data/mixture/icdar_2015/'
test_img_prefix5 = 'data/mixture/svtp/'
test_img_prefix6 = 'data/mixture/ct80/'
test_ann_file1 = 'data/mixture/IIIT5K/test_label.txt'
test_ann_file2 = 'data/mixture/svt/test_label.txt'
test_ann_file3 = 'data/mixture/icdar_2013/test_label_1015.txt'
test_ann_file4 = 'data/mixture/icdar_2015/test_label.txt'
test_ann_file5 = 'data/mixture/svtp/test_label.txt'
test_ann_file6 = 'data/mixture/ct80/test_label.txt'
test1 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/IIIT5K/',
    ann_file='data/mixture/IIIT5K/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test2 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svt/',
    ann_file='data/mixture/svt/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test3 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2013/',
    ann_file='data/mixture/icdar_2013/test_label_1015.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test4 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/icdar_2015/',
    ann_file='data/mixture/icdar_2015/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test5 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/svtp/',
    ann_file='data/mixture/svtp/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
test6 = dict(
    type='OCRDataset',
    img_prefix='data/mixture/ct80/',
    ann_file='data/mixture/ct80/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2011',
                ann_file=('data/mixture/icdar_2011/train_label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=20,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013',
                ann_file=('data/mixture/icdar_2013/train_label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=20,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015',
                ann_file=('data/mixture/icdar_2015/train_label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=20,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/coco_text',
                ann_file=('data/mixture/coco_text/train_label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=20,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/III5K',
                ann_file=('data/mixture/III5K/train_label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=20,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/SynthText_Add',
                ann_file=('data/mixture/SynthText_Add/label.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/SynthText',
                ann_file=('data/mixture/SynthText/shuffle_labels.txt', ),
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/Syn90k',
                ann_file='data/mixture/Syn90k/shuffle_labels.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'text',
                    'valid_ratio'
                ])
        ]),
    val=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=160,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]),
    test=dict(
        type='UniformConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/IIIT5K/',
                ann_file='data/mixture/IIIT5K/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svt/',
                ann_file='data/mixture/svt/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2013/',
                ann_file='data/mixture/icdar_2013/test_label_1015.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/icdar_2015/',
                ann_file='data/mixture/icdar_2015/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/svtp/',
                ann_file='data/mixture/svtp/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True),
            dict(
                type='OCRDataset',
                img_prefix='data/mixture/ct80/',
                ann_file='data/mixture/ct80/test_label.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='LineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiRotateAugOCR',
                rotate_degrees=[0, 90, 270],
                transforms=[
                    dict(
                        type='ResizeOCR',
                        height=48,
                        min_width=48,
                        max_width=160,
                        keep_aspect_ratio=True,
                        width_downsample_ratio=0.25),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'resize_shape',
                            'valid_ratio'
                        ])
                ])
        ]))
evaluation = dict(interval=1, metric='acc')
work_dir = 'seg'
gpu_ids = range(0, 1)
