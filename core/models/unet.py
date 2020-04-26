import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..utils import weights_init

__all__ = ['get_unet']

class UNet(nn.Module):
    def encoder_block(self, in_channels, out_channels, kernel_size=3, dropout=False):
        """
        This function creates one encoder block
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        block = nn.Sequential(*layers)

        return block

    def decoder_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        This function creates one decoder block
        """
        block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)
                )
        return  block

    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                )
        return  block

    def __init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.encode1 = self.encoder_block(in_channel, 64)
        self.encode2 = self.encoder_block(64, 128)
        self.encode3 = self.encoder_block(128, 256)
        self.encode4 = self.encoder_block(256, 512, dropout=True)

        # Bottleneck
        self.bottleneck = self.decoder_block(512, 1024, 512)

        # Decode
        self.decode4 = self.decoder_block(1024, 512, 256)
        self.decode3 = self.decoder_block(512, 256, 128)
        self.decode2 = self.decoder_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        self.__init_weight()

    def forward(self, x):
        # Encode
        encode_block1 = self.encode1(x)
        encode_block2 = self.encode2(encode_block1)
        encode_block3 = self.encode3(encode_block2)
        encode_block4 = self.encode4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.decode4(torch.cat([bottleneck, F.upsample(encode_block4, bottleneck.size()[2:], mode='bilinear')], 1))
        decode_block3 = self.decode3(torch.cat([decode_block4, F.upsample(encode_block3, decode_block4.size()[2:], mode='bilinear')], 1))
        decode_block2 = self.decode2(torch.cat([decode_block3, F.upsample(encode_block2, decode_block3.size()[2:], mode='bilinear')], 1))
        final_layer = self.final_layer(torch.cat([decode_block2, F.upsample(encode_block1, decode_block2.size()[2:], mode='bilinear')], 1))
        
        return F.upsample(final_layer, x.size()[2:], mode='bilinear')

def get_unet(in_channel, out_channel, **kwargs):
    model = UNet(in_channel, out_channel)
    return model