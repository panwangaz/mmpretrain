import numpy as np
from mmengine.config import Config

from mmpretrain.datasets import build_dataset

cfg_path = 'mmpretrain_plugin/configs/datasets/formula.py'
cfg = Config.fromfile(cfg_path)
dataset = build_dataset(cfg.train_dataloader.dataset)

i = 0
while True:
    np.random.seed(0)
    data = dataset.__getitem__(i)
    i = i + 1
