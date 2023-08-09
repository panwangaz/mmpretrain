# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmengine.config import read_base

with read_base():
    from mmpretrain.configs._base_.default_runtime import *
    from mmpretrain.configs._base_.models.resnet18 import *
    from mmpretrain.configs._base_.schedules.imagenet_bs256 import *
    from ..datasets.formula import *

# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, ResNet)

# model settings
model = dict(
    type=ImageClassifier,
    backbone=dict(
        type=ResNet,
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=199,
        in_channels=512,
        loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
        topk=(1, ),
    ))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=32)
