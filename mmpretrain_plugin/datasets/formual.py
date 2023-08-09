# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Optional, Sequence, Union

from mmengine.fileio import get_file_backend

from mmpretrain.datasets.base_dataset import BaseDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class MathsFormula(BaseDataset):
    """A generic dataset for multiple tasks."""

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 metainfo: Optional[dict] = None,
                 lazy_init: bool = False,
                 **kwargs):
        assert (ann_file or data_prefix or data_root), \
            'One of `ann_file`, `data_root` and `data_prefix` must '\
            'be specified.'

        self.extensions = tuple(set([i.lower() for i in extensions]))

        super().__init__(
            # The base class requires string ann_file but this class doesn't
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
            **kwargs)

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def load_data_list(self):
        """Load image paths and gt_labels."""
        with open(self.ann_file, 'r') as f:
            samples = json.load(f)

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        for filename, gt_label in samples.items():
            img_path = backend.join_path(self.img_prefix, filename)
            info = {
                'img_path': img_path,
                'gt_label': self.class_to_idx[int(gt_label)]
            }
            data_list.append(info)

        return data_list
