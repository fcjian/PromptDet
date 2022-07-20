# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core.bbox import bbox_mapping_back, bbox_mapping


@DETECTORS.register_module()
class TwoStagePromptDet(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStagePromptDet, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if gt_bboxes is None:
            return self.forward_self_train(img, img_metas, gt_labels, **kwargs)

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 ignore_novel=True,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def forward_self_train(self,
                            img,
                            img_metas,
                            gt_labels,
                            **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_labels (list[Tensor]): class indices corresponding to each box

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # two-scale labeling
        img_metas2 = kwargs.pop('img_metas2')
        img2 = kwargs.pop('img2')
        gt_labels2 = kwargs.pop('gt_labels2')

        x, gt_bboxes, gt_labels, gt_scores = self.get_feature_and_pseudo_label(img, img_metas, gt_labels)
        x2, gt_bboxes2, gt_labels2, gt_scores2 = self.get_feature_and_pseudo_label(img2, img_metas2, gt_labels2)

        ori_gt_bboxes = self.bbox_mapping_back_batch(gt_bboxes, img_metas)
        ori_gt_bboxes2 = self.bbox_mapping_back_batch(gt_bboxes2, img_metas2)
        gt_bboxes1to2 = self.bbox_mapping_batch(ori_gt_bboxes, img_metas2)
        gt_bboxes2to1 = self.bbox_mapping_batch(ori_gt_bboxes2, img_metas)

        gt_bboxes = list(gt_bboxes)
        gt_bboxes2 = list(gt_bboxes2)
        for i in range(len(gt_bboxes)):
            if gt_scores[i][0] > gt_scores2[i][0]:
                gt_bboxes2[i] = gt_bboxes1to2[i]
            else:
                gt_bboxes[i] = gt_bboxes2to1[i]
        gt_labels = tuple(gt_labels)
        gt_labels2 = tuple(gt_labels2)

        # train
        losses = self.forward_train_for_uncurated_images(x,  img_metas, gt_bboxes, gt_labels, **kwargs)
        losses2 = self.forward_train_for_uncurated_images(x2, img_metas2, gt_bboxes2, gt_labels2, **kwargs)

        losses_uncurated = dict()
        for key in losses:
            if isinstance(losses[key], list):
                losses_uncurated[key] = []
                for i in range(len(losses[key])):
                    if key == 'loss_cls_uncurated':
                        losses_uncurated[key].append(losses[key][i] + losses2[key][i])
                    else:
                        losses_uncurated[key].append((losses[key][i] + losses2[key][i]) / 2.0)
            else:
                if key == 'loss_cls_uncurated':
                    losses_uncurated[key] = losses[key] + losses2[key]
                else:
                    losses_uncurated[key] = (losses[key] + losses2[key]) / 2.0

        return losses_uncurated

    def forward_train_for_uncurated_images(self,
                                            x,
                                            img_metas,
                                            gt_bboxes,
                                            gt_labels,
                                            rpn_ignore_neg=True,
                                            rcnn_ignore_neg=True,
                                            **kwargs):
        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg,
            ignore_neg=rpn_ignore_neg)

        for key in rpn_losses:
            # train rpn classifier
            if key == 'loss_rpn_cls':
                losses[key + '_uncurated'] = rpn_losses[key]

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore=None, gt_masks=None,
                                                 ignore_neg=rcnn_ignore_neg, **kwargs)

        for key in roi_losses:
            losses[key + '_uncurated'] = roi_losses[key]

        return losses

    def bbox_mapping_back_batch(self, batch_bboxes, img_metas):
        recovered_bboxes = []
        for bboxes, img_info in zip(batch_bboxes, img_metas):
            img_shape = img_info['img_shape']
            scale_factor = img_info['scale_factor']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        return recovered_bboxes

    def bbox_mapping_batch(self, batch_bboxes, img_metas):
        recovered_bboxes = []
        for bboxes, img_info in zip(batch_bboxes, img_metas):
            img_shape = img_info['img_shape']
            scale_factor = img_info['scale_factor']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            bboxes = bbox_mapping(bboxes, img_shape, scale_factor, flip,
                                  flip_direction)
            recovered_bboxes.append(bboxes)
        return recovered_bboxes

    def get_feature_and_pseudo_label(self,
                                    img,
                                    img_metas,
                                    gt_labels,
                                    ):
        x = self.extract_feat(img)

        # RPN forward and loss
        with torch.no_grad():
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_proposals(x, img_metas, proposal_cfg=proposal_cfg)

            topK_proposals = self.train_cfg.get('pseudo_label', {}).get('topK_proposals', 20)
            gt_bboxes, gt_labels, gt_scores = \
                self.roi_head.get_pseudo_label_with_score(x, proposal_list, gt_labels, topK_proposals)

        return x, gt_bboxes, gt_labels, gt_scores

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
