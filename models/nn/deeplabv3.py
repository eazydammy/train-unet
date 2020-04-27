from torch import nn
from backbone import get_backbone
from torch.nn import functional as F
import torch
from datasets import datasets
from torchviz import make_dot
from models.components.norm import get_norm

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, up_method, norm='bn'):
        super(ASPPPooling, self).__init__()
        self.up_method = up_method
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                 get_norm(norm, channels=out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        image_feature = self.global_pooling(x)
        return F.interpolate(image_feature, (h, w), **self.up_method)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, up_method, rate=1, norm='bn'):
        super(ASPP, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            get_norm(norm, channels=out_channels),
            nn.ReLU(True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=6 * rate, padding=6 * rate,
                      bias=False),
            get_norm(norm, channels=out_channels),
            nn.ReLU(True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=12 * rate, padding=12 * rate,
                      bias=False),
            get_norm(norm, channels=out_channels),
            nn.ReLU(True)
        )

        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=18 * rate, padding=18 * rate,
                      bias=False),
            get_norm(norm, channels=out_channels),
            nn.ReLU(True)
        )
        self.up_method = up_method
        self.pooling_branch = ASPPPooling(in_channels, out_channels, self.up_method)

        self.merge_branch = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            get_norm(norm, channels=out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False)
        )

    def forward(self, x):
        b1 = self.branch_1(x)
        b2 = self.branch_2(x)
        b3 = self.branch_3(x)
        b4 = self.branch_4(x)
        image_feature = self.pooling_branch(x)
        y = torch.cat((b1, b2, b3, b4, image_feature), 1)
        return self.merge_branch(y)


class DeepLabV3Core(nn.Module):
    def __init__(self, in_channels, out_channels, up_method, os=16, norm='bn'):
        super(DeepLabV3Core, self).__init__()
        rate = 16 // os
        inter_channels = in_channels // os
        self.up_method = up_method
        self.aspp = ASPP(in_channels, inter_channels, self.up_method, rate=rate)
        self.tail = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm, channels=inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        x = self.tail(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, nbr_classes, backbone='xception', deep_supervision=True, os=16, norm='bn', **kwargs):
        super(DeepLabV3, self).__init__()
        self.nbr_classes = nbr_classes
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.backbone = get_backbone(backbone, norm=norm, **kwargs)
        self.core = DeepLabV3Core(in_channels=2048, out_channels=nbr_classes, up_method=self.up_method,
                                  os=os, norm=norm)
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

        x = self.core(x)
        x = F.interpolate(x, (h, w), **self.up_method)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            return x, aux
        return x

def get_deeplabv3(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', norm='bn', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    deeplab = DeepLabV3(nbr_classes, deep_supervision=supervision, backbone=backbone, norm=norm, **kwargs)
    if model_pretrained:
        deeplab.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return deeplab

if __name__ == '__main__':
    model = get_deeplabv3(backbone='xception', model_pretrained=False, backbone_pretrained=False, os=16)
    g = make_dot(model(torch.rand(16, 3, 384, 384)), params=dict(model.named_parameters()))
    g.render('deeplabv3')

    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("layer architecture：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("the number of parameters：" + str(l))
    #     k = k + l
    # print("the total of parameters：" + str(k))