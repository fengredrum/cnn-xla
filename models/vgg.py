import torch
import torch.nn as nn
import numpy as np
from . import Swish, Mish


class VGGBlock(nn.Module):
    def __init__(self,
                 num_convs,
                 in_channels,
                 out_channels,
                 activation='relu'):
        super(VGGBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        net.append(self.activation)
        if num_convs > 1:
            for _ in range(1, num_convs):
                net.append(
                    nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1))
                net.append(self.activation)
        net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class VGG(nn.Module):
    def __init__(self,
                 conv_arch,
                 image_size=224,
                 num_hiddens=4096,
                 activation='relu',
                 num_classes=10):
        super(VGG, self).__init__()
        self.image_size = image_size

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        self.conv = nn.Sequential()
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module(
                'vgg_block_' + str(i),
                VGGBlock(num_convs, in_channels, out_channels, activation))

        feature_size = self._get_feature_size(self.conv)
        print('feature: ', feature_size)
        self.linear = nn.Sequential(
            nn.Linear(512 * np.prod(feature_size), num_hiddens),
            self.activation, nn.Dropout(0.5),
            nn.Linear(num_hiddens, num_hiddens), self.activation,
            nn.Dropout(0.5), nn.Linear(num_hiddens, 10))

    def _get_feature_size(self, net):
        x = torch.zeros(1, 3, self.image_size, self.image_size)
        out = net(x)
        return out.size()[2:]

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        return self.linear(out)


if __name__ == "__main__":
    conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512),
                 (2, 512, 512))

    image_size = 32
    x = torch.zeros(1, 3, image_size, image_size)
    net = VGG(conv_arch, image_size)
    out = net(x)
    print(out.shape)
