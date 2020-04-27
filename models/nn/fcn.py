import torch.nn as nn
from backbone import get_backbone
from torch.nn import functional as F
import torch
from datasets import datasets
from models.components.norm import get_norm
import numpy as np
from torchviz import make_dot

"""
    Reference:
        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." CVPR, 2015
"""

def fcn_conv(in_channels, out_channels, norm):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        get_norm(norm, channels=in_channels),
        nn.ReLU(False),
        nn.Dropout2d(0.1, False),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    )

class FCNCore(nn.Module):
    def __init__(self, out_channels, skip_dims, ratio_mapping, norm='bn', S=8):
        super(FCNCore, self).__init__()

        if S not in [8, 16, 32]:
            raise RuntimeError(" `S` can only be 8, 16 or 32, but S={}".format(S))

        if S not in ratio_mapping:
            raise RuntimeError("`S`={} is not supported by the current backbone".format(S))

        self.S = S
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.ratio_mapping = ratio_mapping

        if S == 32 and 32 in ratio_mapping:
            self.conv = fcn_conv(skip_dims[ratio_mapping[32]], out_channels, norm)
        elif S == 16:
            if 32 in ratio_mapping:
                self.conv_32s2cls = nn.Conv2d(skip_dims[ratio_mapping[32]], out_channels, 1)
            self.conv_16s2cls = fcn_conv(skip_dims[ratio_mapping[16]], out_channels, norm)
        elif S == 8:
            if 32 in ratio_mapping:
                self.conv_32s2cls = nn.Conv2d(skip_dims[ratio_mapping[32]], out_channels, 1)
            if 16 in ratio_mapping:
                self.conv_16s2cls = nn.Conv2d(skip_dims[ratio_mapping[16]], out_channels, 1)
            self.conv_8s2cls = fcn_conv(skip_dims[ratio_mapping[8]], out_channels, norm)

    def forward(self, x, skip_connections, skip_dims):

        if self.S == 32:
            x = self.conv(x)
        elif self.S == 16:
            if 32 in self.ratio_mapping:
                x = self.conv_32s2cls(x)
                _, _, h16, w16 = skip_connections[self.ratio_mapping[16]].size()
                x = F.interpolate(x, (h16, w16), **self.up_method)
                x = x + self.conv_16s2cls(skip_connections[self.ratio_mapping[16]])
            else:
                x = self.conv_16s2cls(skip_connections[self.ratio_mapping[16]])
        elif self.S == 8:
            if 16 in self.ratio_mapping:
                if 32 in self.ratio_mapping:
                    _, _, h16, w16 = skip_connections[self.ratio_mapping[16]].size()
                    x = self.conv_32s2cls(x)
                    x = F.interpolate(x, (h16, w16), **self.up_method)
                    x = x + self.conv_16s2cls(skip_connections[self.ratio_mapping[16]])
                else:
                    x = self.conv_16s2cls(skip_connections[self.ratio_mapping[16]])
                _, _, h8, w8 = skip_connections[self.ratio_mapping[8]].size()
                x = F.interpolate(x, (h8, w8), **self.up_method)
                skip8s = self.conv_8s2cls(skip_connections[self.ratio_mapping[8]])
                x = skip8s + x
            else:
                x = self.conv_8s2cls(skip_connections[self.ratio_mapping[8]])
        return x


class FCN(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50', norm='bn', S=8, **kwargs):
        super(FCN, self).__init__()
        self.nbr_classes = nbr_classes
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.deep_supervision = deep_supervision
        self.backbone = get_backbone(backbone, norm=norm, **kwargs)
        self.core = FCNCore(out_channels=nbr_classes, skip_dims=self.backbone.skip_dims,
                            ratio_mapping=self.backbone.ratio_mapping,
                            norm=norm, S=S)
        self.S = S
        if deep_supervision:
            self.aux_branch = fcn_conv(self.backbone.aux_dim, nbr_classes, norm)

    def get_parameters_as_groups(self, lr, different_lr_in_layers=True):
        parameters = []
        if different_lr_in_layers:
            parameters.append({'params': self.backbone.parameters(), 'lr':lr})
            parameters.append({'params': self.core.parameters(), 'lr':lr*10})
            if self.deep_supervision:
                parameters.append({'params': self.aux_branch.parameters(), 'lr': lr * 10})
        else:
            parameters.append({'params': self.parameters(), 'lr': lr })
        return parameters

    def forward(self, x):
        _, _, h, w = x.size()
        x, aux = self.backbone.backbone_forward(x)
        x = self.core(x, self.backbone.skip_connections, self.backbone.skip_dims)
        x = F.interpolate(x, (h, w), **self.up_method)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            return x, aux
        return x

def get_fcn(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', norm='bn', S=32, **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    fcn = FCN(nbr_classes, supervision, backbone, norm=norm, S=S, **kwargs)
    if model_pretrained:
        fcn.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return fcn

if __name__ == '__main__':
    model = get_fcn(backbone='vgg19bn', model_pretrained=False, backbone_pretrained=False, sk_conn=True, S=8)
    g = make_dot(model(torch.rand(16, 3, 256, 256)), params=dict(model.named_parameters()))
    g.render('fcn')
    model(torch.rand(16, 3, 256, 256))
