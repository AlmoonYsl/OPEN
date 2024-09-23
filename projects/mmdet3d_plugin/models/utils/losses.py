# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models import LOSSES


def depth_focal_loss(pred,
                    target,
                    weight=None,
                    gamma=2.0,
                    alpha=0.25,
                    reduction='mean',
                    avg_factor=None):

    assert len(target) == 2
    assert gamma == 2.0
    label, score = target

    pred_sigmoid = pred.sigmoid()
    onehot = torch.zeros((pred_sigmoid.shape[0], pred.shape[1] + 1)).to(pred)
    onehot.scatter_(1, label[:, None], 1)
    onehot = onehot[:, 0:-1]

    soft_label = (onehot > 0).float() * score[:, None]
    pt = soft_label - pred_sigmoid
    focal_weight = ((1 - alpha) + (2 * alpha - 1) * soft_label) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, soft_label, reduction='none') * focal_weight

    weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class DepthFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(DepthFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * depth_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls