from torch import nn
from backbone import get_backbone
from torch.nn import functional as F
import torch
from datasets import datasets
from models.nn.deeplabv3 import ASPP
from torchviz import make_dot
from models.components.norm import get_norm

class DeepLabV3PlusCore(nn.Module):
    def __init__(self, in_channels, out_channels, backbone,up_method, os=16, norm='bn'):
        super(DeepLabV3PlusCore, self).__init__()
        rate = 16 // os
        inter_channels = in_channels // os
        self.up_method = up_method
        self.aspp = ASPP(in_channels, inter_channels, self.up_method, rate=rate)
        self.backbone = backbone
        self.outer_branch = nn.Sequential(
            nn.Conv2d(256, 48, 1, 1, padding=0, bias=False),
            get_norm(norm, channels=48),
            nn.ReLU(inplace=True)
        )
        self.os = os
        self.merge_layers = nn.Sequential(
            nn.Conv2d(inter_channels + 48, inter_channels, 3, 1, padding=1, bias=False),
            get_norm(norm, channels=inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, 3, 1, padding=1, bias=False),
            get_norm(norm, channels=inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1, False)
        )

        self.cls = nn.Conv2d(inter_channels, out_channels, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        x, aux = self.backbone.backbone_forward(x)

        x = self.aspp(x)
        # x = self.dropout(x)
        x = F.interpolate(x, (h//4, w//4), **self.up_method)

        hook = self.backbone.outer_branches[0]
        hook = self.outer_branch(hook)
        x = torch.cat([x, hook], 1)
        x = self.merge_layers(x)
        x = self.cls(x)
        x = F.interpolate(x, (h, w), **self.up_method)
        return x, aux


class DeepLabV3Plus(nn.Module):
    def __init__(self, nbr_classes, backbone='xception', deep_supervision=True, os=16, norm='bn', **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.nbr_classes = nbr_classes
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.backbone = get_backbone(backbone, norm=norm, **kwargs)
        self.core = DeepLabV3PlusCore(in_channels=2048, out_channels=nbr_classes, backbone=self.backbone, up_method=self.up_method, os=os)
        if deep_supervision:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
                get_norm(norm, channels=256),
                nn.ReLU(False),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(256, nbr_classes, kernel_size=1, stride=1)
            )
        self.deep_supervision = deep_supervision

    def get_parameters_as_groups(self, lr, different_lr_in_layers=True):
        parameters = []
        if different_lr_in_layers:
            parameters.append({'params': self.core.backbone.parameters(), 'lr':lr})
            parameters.append({'params': self.core.aspp.parameters(), 'lr':lr*10})
            parameters.append({'params': self.core.outer_branch.parameters(), 'lr':lr*10})
            parameters.append({'params': self.core.merge_layers.parameters(), 'lr':lr*10})
            parameters.append({'params': self.core.cls.parameters(), 'lr':lr*10})
            if self.deep_supervision:
                parameters.append({'params': self.aux_branch.parameters(), 'lr': lr * 10})
        else:
            parameters.append({'params': self.parameters(), 'lr': lr })
        return parameters

    def forward(self, x):
        _, _, h, w = x.size()
        x, aux = self.core(x)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            return x, aux
        return x

def get_deeplabv3plus(backbone='xception', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    deeplab = DeepLabV3Plus(nbr_classes, deep_supervision=supervision, backbone=backbone, **kwargs)
    if model_pretrained:
        deeplab.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return deeplab

if __name__ == '__main__':
    model = get_deeplabv3plus(backbone='xception', model_pretrained=False, backbone_pretrained=False, os=8)
    g = make_dot(model(torch.rand(16, 3, 384, 384)), params=dict(model.named_parameters()))
    g.render('deeplabv3plus_res50')