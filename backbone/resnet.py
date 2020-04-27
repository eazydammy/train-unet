import torch
import torch.nn as nn
import math
from models.components.norm import get_norm
"""
    Reference:
        [1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        [2] Yu, Fisher , and V. Koltun . "Multi-Scale Context Aggregation by Dilated Convolutions." (2015).
        [3] Yu, Fisher , V. Koltun , and T. Funkhouser . "Dilated Residual Networks." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) IEEE Computer Society, 2017.
"""

net_structures = {50: [3, 4, 6, 3],
                  101: [3, 4, 23, 3],
                  152: [3, 8, 36, 3]}

def conv3x3(in_channel, out_channel, stride=1):
    """ 3x3 convolution layer """
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    """ Basic residual block for Dilated Residual Networks (DRN) """
    def __init__(self, in_channel, out_channel, stride=1, dilation=1,
                 downsample=None, residual=True, norm='bn'):
        super(BasicBlock, self).__init__()
        self.conv_1_3x3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=dilation,
                                    dilation=dilation, bias=False)
        self.bn_1 = get_norm(norm, channels=out_channel)

        self.conv_2_3x3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=dilation,
                                    dilation=dilation, bias=False)
        self.bn_2 = get_norm(norm, channels=out_channel)

        self.downsample = downsample
        self.residual = residual
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv_1_3x3(x)
        out = self.relu(self.bn_1(out))

        out = self.conv_2_3x3(out)
        out = self.relu(self.bn_2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual

        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    """ Bottleneck for Dilated Residual Networks (DRN) """
    def __init__(self, in_channel, out_channel, stride=1, dilation=1,
                 downsample=None, residual=True, norm='bn'):
        super(Bottleneck, self).__init__()
        # 1x1
        self.conv_1_1x1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn_1 = get_norm(norm, channels=out_channel)

        # 3x3
        self.conv_2_3x3 = nn.Conv2d(out_channel, out_channel, 3, stride=stride, padding=dilation, dilation=dilation,
                                    bias=False)
        self.bn_2 = get_norm(norm, channels=out_channel)

        # 1x1
        self.conv_3_1x1 = nn.Conv2d(out_channel, 4 * out_channel, 1, bias=False)
        self.bn_3 = get_norm(norm, channels=4 * out_channel)

        self.downsample = downsample
        self.residual = residual
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.conv_1_1x1(x)
        out = self.relu(self.bn_1(out))

        out = self.conv_2_3x3(out)
        out = self.relu(self.bn_2(out))

        out = self.conv_3_1x1(out)
        out = self.bn_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual

        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, nbr_classes=1000, is_bottleneck = True, nbr_layers=50, sk_conn=False, norm='bn'):
        super(ResNet, self).__init__()
        global net_structures
        if nbr_layers not in net_structures:
            raise RuntimeError("nbr_layers can only be 50, 101 or 152, but got {}".format(nbr_layers))
        net_structure = net_structures[nbr_layers]
        self.sk_conn = sk_conn
        self.aux_dim = 1024
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            get_norm(norm, channels=64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            conv3x3(64, 64),
            get_norm(norm, channels=64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            conv3x3(64, 128),
            get_norm(norm, channels=128),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        residual_block = Bottleneck if is_bottleneck else BasicBlock

        self.block_1 = self._make_layers(residual_block, 128, 64, net_structure[0], norm=norm)
        self.block_2 = self._make_layers(residual_block, 64 * residual_block.expansion, 128, net_structure[1],
                                         stride=2, norm=norm)
        self.block_3 = self._make_layers(residual_block, 128 * residual_block.expansion, 256, net_structure[2],
                                         dilation=2, norm=norm)
        self.block_4 = self._make_layers(residual_block, 256 * residual_block.expansion, 512, net_structure[3],
                                         dilation=4, norm=norm)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * residual_block.expansion, nbr_classes)

        if self.sk_conn:
            self.skip_dims = [2048, 1024, 256, 128]
            self.ratio_mapping = {2: -1, 4: -2, 8: -3}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        print("init resnet")

    def base_forward(self, x):
        self.outer_branches = []
        if self.sk_conn:
            self.skip_connections = []
        # 3, h, w
        x = self.conv1(x)
        # 64, /2, /2
        x = self.conv2(x)
        # 64, /2, /2
        x = self.conv3(x)
        # 128, /2, /2
        if self.sk_conn:
            self.skip_connections.append(x)
        x = self.maxpool(x)
        # 128, /4, /4
        x = self.block_1(x)
        # 256, /4, /4
        if self.sk_conn:
            self.skip_connections.append(x)
        self.outer_branches.append(x)
        x = self.block_2(x)
        # 512, /8, /8
        x = self.block_3(x)
        # 1024, /8, /8
        if self.sk_conn:
            self.skip_connections.append(x)
        return x

    def forward(self, x):
        x = self.base_forward(x)
        x = self.block_4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def backbone_forward(self, x):
        aux = self.base_forward(x)
        x = self.block_4(aux)
        if self.sk_conn:
            self.skip_connections.reverse()
        return x, aux

    def _make_layers(self, block, in_channel, out_channel, nbr_blocks, stride=1, dilation=1, norm='bn'):
        downsample = None
        if stride != 1 or in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                get_norm(norm, channels=out_channel * block.expansion)
            )
        layers = []
        layers.append(block(in_channel, out_channel, stride=stride, downsample=downsample,
            dilation=1 if dilation == 1 else dilation // 2, norm=norm))

        in_channel = out_channel * block.expansion

        for _ in range(1, nbr_blocks):
            layers.append(block(in_channel, out_channel, dilation=dilation, norm=norm))

        return nn.Sequential(*layers)

def get_resnet(nbr_layers=50):

    def build_net(backbone_pretrained_path='../weights/resnet50.pth', nbr_classes=1000, is_bottleneck = True,
                  backbone_pretrained=True, norm='bn', sk_conn = False, **kwargs):
        model = ResNet(nbr_classes, is_bottleneck, nbr_layers, norm=norm, sk_conn=sk_conn)
        if backbone_pretrained:
            pretrain_weights = torch.load(backbone_pretrained_path)
            pretrained_weights_list = list(pretrain_weights.items())

            model_weights = model.state_dict()
            count = 0
            layer_success = 0
            for k, v in model_weights.items():
                pretrain_k, pretrain_v = pretrained_weights_list[count]
                if pretrain_v.shape == v.shape:
                    model_weights[k] = pretrain_v
                    layer_success += 1
                count += 1
            model_weights.update(model_weights)
            model.load_state_dict(model_weights, strict=False)
            print('resnet%d weights are loaded successfully: %d/%d' % (nbr_layers, layer_success, len(pretrain_weights)))

        return model

    return build_net

if __name__ == '__main__':
    sample = torch.rand(16, 3, 256, 256)
    model = get_resnet(50)(backbone_pretrained=True, is_bottleneck=True, sk_conn=True)
    model(sample)
    print(len(model.skip_connections))