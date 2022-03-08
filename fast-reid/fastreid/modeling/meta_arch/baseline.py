# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'fastavgpool':  pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':    pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)
        self.extra_feat_dim = cfg.MODEL.HEADS.EXTRA_FEAT

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        mean_std_feat = None
        if self.extra_feat_dim > 0:
            mean_std_feat = self.cal_mean_std(images)
            mean_std_feat = mean_std_feat.view(mean_std_feat.size(0), -1, 1, 1)
            assert mean_std_feat.size(1) == self.extra_feat_dim, 'wrong extra features! Config: {}, Actual: {}'.format(mean_std_feat.size(1), self.extra_feat_dim)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            return self.heads(features, targets, extra_feat=mean_std_feat), targets
        else:
            return self.heads(features, extra_feat=mean_std_feat)

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)

        # if isinstance(batched_inputs, dict):
        #     images = batched_inputs["images"]
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def cal_mean_std(self, images):
        """
        Calculate mean and std of R, G, B as extra features
        """
        mean = torch.mean(images, dim=(2, 3))
        std = torch.std(images, dim=(2, 3))
        return torch.cat((mean, std), dim=-1)

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # cls_outputs, pred_class_logits, pred_features = outputs
        cls_outputs = outputs['cls']
        pred_class_logits = outputs['logits']
        pred_features = outputs['feat']
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = CrossEntropyLoss(self._cfg)(cls_outputs, gt_labels)
            # Log prediction accuracy
            CrossEntropyLoss.log_accuracy(pred_class_logits.detach(), gt_labels)

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = TripletLoss(self._cfg)(pred_features, gt_labels)

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = torch.mean(CircleLoss(self._cfg)(pred_features, gt_labels))

        if "NpairLoss" in loss_names:
            loss_dict['loss_npair'] = NpairLoss(self._cfg)(pred_features, gt_labels)

        return loss_dict
