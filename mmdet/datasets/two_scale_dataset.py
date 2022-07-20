# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset
import os
import time
import random


@DATASETS.register_module()
class TwoScaleDataset(CocoDataset):
    def __init__(self, min_scales, max_scale, **kwargs):
        self.min_scales = min_scales
        self.max_scale = max_scale
        print(f'using two scale: {self.min_scales}')
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            scale = random.randint(self.min_scales[0], self.min_scales[1])
            data = self.prepare_train_img(idx, scale)
            scale2 = random.randint(self.min_scales[1], self.min_scales[2])
            data2 = self.prepare_train_img(idx, scale2)
            for key in data2:
                data[f'{key}2'] = data2[key]
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx, scale=None):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        
        if scale is not None:
            results['scale'] = (self.max_scale, scale)

        return self.pipeline(results)
