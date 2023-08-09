# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from ..datasets.formula import *
    from mmpretrain.configs._base_.default_runtime import *
    from mmpretrain.configs._base_.models.convnext_base import *
    from mmpretrain.configs._base_.schedules.imagenet_bs1024_adamw_swin import *

from mmengine.model import TruncNormalInit

from mmpretrain.engine import EMAHook
from mmpretrain.models import (ConvNeXt, CutMix, ImageClassifier,
                               LabelSmoothLoss, LinearClsHead, Mixup)

# dataset setting
train_dataloader.update(batch_size=32)

# Model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(type=ConvNeXt, arch='base', drop_path_rate=0.5),
    head=dict(
        type=LinearClsHead,
        num_classes=199,
        in_channels=1024,
        loss=dict(type=LabelSmoothLoss, label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type=TruncNormalInit, layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    train_cfg=dict(augments=[
        dict(type=Mixup, alpha=0.8),
        dict(type=CutMix, alpha=1.0),
    ]),
)

# schedule setting
optim_wrapper.update(
    optimizer=dict(lr=4e-3),
    clip_grad=dict(max_norm=5.0),
)

# runtime setting
custom_hooks = [dict(type=EMAHook, momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
