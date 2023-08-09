# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.dataset import DefaultSampler

from mmpretrain.datasets import LoadImageFromFile, PackInputs, RandomFlip
from mmpretrain.evaluation import Accuracy
from mmpretrain_plugin.datasets import MathsFormula

# dataset settings
dataset_type = MathsFormula
data_preprocessor = dict(
    num_classes=199,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomFlip, prob=0.5, direction='horizontal'),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type=dataset_type,
        data_root='data/sft',
        ann_file='labels/train_label.json',
        data_prefix='train',
        classes=[i for i in range(-99, 100)],
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/sft',
        ann_file='labels/val_label.json',
        data_prefix='val',
        classes=[i for i in range(-99, 100)],
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy)

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
