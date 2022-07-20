# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage_promptdet import TwoStagePromptDet


@DETECTORS.register_module()
class PromptDet(TwoStagePromptDet):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(PromptDet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
