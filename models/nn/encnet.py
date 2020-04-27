import torch.nn as nn
from datasets import datasets
import torch
from backbone import get_backbone
from models.components.encoding import Encoding
import torch.nn.functional as F
from torchviz import make_dot
from models.components.norm import get_norm

class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)

class EncCore(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss, dim_codes, norm='bn'):
        super(EncCore, self).__init__()
        self.top = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            get_norm(norm, channels=512),
            nn.ReLU(inplace=True)
        )
        self.encoding = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            get_norm(norm, channels=512),
            nn.ReLU(inplace=True),
            Encoding(D=512, K=dim_codes),
            get_norm('{}1d'.format(norm) if norm == 'bn' or norm == 'sn' else 'gn', channels=dim_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.se_loss = se_loss
        if se_loss:
            self.se_layer = nn.Linear(512, out_channels)

        self.tail = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(512, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.top(x)
        code = self.encoding(x)
        b, c, _, _ = x.size()
        g = self.fc(code)
        y = g.view(b, c, 1, 1)
        outputs = [F.relu(x + x * y, True)]
        if self.se_loss:
            outputs.append(self.se_layer(code))
        outputs[0] = self.tail(outputs[0])
        return tuple(outputs)


class EncNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50', se_loss=True, norm='bn', **kwargs):
        super(EncNet, self).__init__()
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.nbr_classes = nbr_classes
        self.backbone = get_backbone(backbone, norm=norm, **kwargs)
        self.core = EncCore(in_channels=2048, out_channels=nbr_classes, se_loss=se_loss, dim_codes=32, norm=norm)
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
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

        x = list(self.core(x))
        x[0] = F.interpolate(x[0], (h, w), **self.up_method)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            x.append(aux)
        return tuple(x)

def get_encnet(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', norm='bn', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    enc = EncNet(nbr_classes, supervision, backbone, norm=norm, **kwargs)
    if model_pretrained:
        enc.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return enc

if __name__ == '__main__':
    model = get_encnet(model_pretrained=False, backbone_pretrained=False)
    g = make_dot(model(torch.rand(16, 3, 384, 384)), params=dict(model.named_parameters()))
    # g.format = 'jpg'
    g.render('encnet')
    # viz.image(
    #     cv2.imread(path).swapaxes(1, 2).swapaxes(0, 1),
    #     opts=dict(title='encnet Arch', caption='encnet'),
    # )
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("layer architecture：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("the number of parameters：" + str(l))
        k = k + l
    print("the total of parameters：" + str(k))