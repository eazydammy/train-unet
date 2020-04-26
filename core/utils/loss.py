"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    # def __init__(self, ignore_index=-1, **kwargs):
    def __init__(self, ignore_index=-1):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)

    # def forward(self, *inputs, **kwargs):
    def forward(self, preds, target):
        # preds, target = tuple(inputs)
        # inputs = tuple(list(preds) + [target])

        # return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))
        return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(preds, target))

def get_segmentation_loss(**kwargs):
    return MixSoftmaxCrossEntropyLoss(**kwargs)