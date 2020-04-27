import torch.nn as nn
from backbone import get_backbone
from torch.nn import functional as F
import torch
from datasets import datasets
from models.components.norm import get_norm
from torchviz import make_dot

"""
    Reference:
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
"""

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', up_method='conv'):
        super(Up, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if up_method == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x2, concat=True, is_up=True):

        if is_up:
            x = self.up(x)

        if concat:
            _, _, h2, w2 = x2.size()
            _, _, h1, w1 = x.size()
            diffH = h2 - h1
            diffW = w2 - w1
            x = F.pad(x, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2))
            x = torch.cat([x, x2], dim=1)

        if concat:
            x = self.conv_up(x)
        else:
            x = self.conv(x)

        return x

class UNetCore(nn.Module):
    def __init__(self, out_channels, norm='bn', up_method='conv', skip_dims=None):
        super(UNetCore, self).__init__()
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.ups = nn.ModuleList([])
        for i in range(1, len(skip_dims)):
            self.ups.append(Up(skip_dims[i - 1], skip_dims[i], up_method=up_method, norm=norm))
        self.ups.append(Up(skip_dims[i], skip_dims[i] // 2, up_method=up_method, norm=norm))

        self.conv = nn.Sequential(
            nn.Conv2d(skip_dims[-1] // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def __up(self, up, skip, x, conn_idx):
        if x.shape[2] * 2 == skip.shape[2] and x.shape[3] * 2 == skip.shape[3] and x.shape[1] == skip.shape[1]:
            x = up(x, skip)
            conn_idx += 1
        elif x.shape[2] * 2 == skip.shape[2] and x.shape[3] * 2 == skip.shape[3] and x.shape[1] != skip.shape[1]:
            x = up(x, None, False, False)
        else:
            x = up(x, None, False, False)
            conn_idx += 1
        return x, conn_idx

    def forward(self, x, skip_connections, skip_dims):
        _, _, h, w = x.size()
        conn_idx = 0
        for up in self.ups:
            x, conn_idx = self.__up(up, skip_connections[conn_idx], x, conn_idx)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, nbr_classes, backbone='vgg16', norm='bn', deep_supervision=True, **kwargs):
        super(UNet, self).__init__()
        self.nbr_classes = nbr_classes
        self.deep_supervision = deep_supervision
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        up_method = 'conv'
        self.backbone = get_backbone(backbone, **kwargs)
        self.core = UNetCore(out_channels=nbr_classes,
                             norm=norm, up_method=up_method, skip_dims = self.backbone.skip_dims)

        if deep_supervision:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(self.backbone.aux_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
                get_norm(norm, channels=256),
                nn.ReLU(False),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(256, nbr_classes, kernel_size=1, stride=1)
            )

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
        _, _, h2, w2 = x.size()
        if h2 != h or w2 != w:
            x = F.interpolate(x, (h, w), **self.up_method)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            return x, aux
        return x

def get_unet(backbone='vgg16', model_pretrained=True,
               model_pretrain_path=None, dataset='ade20k', norm='bn', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    psp = UNet(nbr_classes, backbone, norm=norm, **kwargs)
    if model_pretrained:
        psp.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return psp

if __name__ == '__main__':
    model = get_unet(backbone='resnet50', model_pretrained=False, backbone_pretrained=False, deep_supervision=True,
                     sk_conn=True)
    g = make_dot(model(torch.rand(16, 3, 256, 256)), params=dict(model.named_parameters()))
    g.render('unet_resnet50')