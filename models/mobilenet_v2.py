'''
A PyTorch implementation of MobileNetV2.
The original paper can be found at https://arxiv.org/abs/1801.04381.
'''

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import activetion_func


class InvBottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 expansion_ratio=6,
                 activation='relu6'):
        super(InvBottleneckBlock, self).__init__()
        self.activation = activetion_func(activation)
        inner_channels = expansion_ratio * in_channels

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels), self.activation,
            nn.Conv2d(inner_channels,
                      inner_channels,
                      kernel_size=3,
                      padding=1,
                      stride=stride,
                      groups=inner_channels,
                      bias=False), nn.BatchNorm2d(inner_channels),
            self.activation,
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        elif stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.trunk(x)
        if self.shortcut:
            out += self.shortcut(x)
        return out


class MobileNetV2(nn.Module):
    def __init__(self,
                 activation='relu',
                 num_classes=10,
                 width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        self.activation = activetion_func(activation)
        self.architecture = np.asarray(
            [[32, 1, 2], [16, 1, 1], [24, 2, 1], [32, 3, 2], [64, 4, 2],
             [96, 3, 1], [160, 3, 2], [320, 1, 1]],
            dtype=np.float32)
        self.architecture[:, 0] *= width_multiplier
        self.architecture = np.floor(self.architecture).astype(
            np.int32).tolist()

        self.num_channels = self.architecture[0][0]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.num_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.num_channels), self.activation,
            InvBottleneckBlock(self.num_channels,
                               self.architecture[1][0],
                               expansion_ratio=1,
                               activation=activation))
        self.num_channels = self.architecture[1][0]
        self.layer2 = self._make_layers(activation)
        out_channels = math.floor(1280 * width_multiplier)
        self.conv1 = nn.Conv2d(self.num_channels,
                               out_channels,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               num_classes,
                               kernel_size=1,
                               bias=False)

    def _make_layers(self, activation):
        net = []
        for i, (c, n, s) in enumerate(self.architecture):
            if i < 2:
                continue

            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layer = InvBottleneckBlock(in_channels=self.num_channels,
                                           out_channels=c,
                                           stride=stride,
                                           activation=activation)

                net.append(layer)
                self.num_channels = c
        return nn.Sequential(*net)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.activation(self.bn1(self.conv1(out)))
        out = F.avg_pool2d(out, 4)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        return out


def mobilenet_v2(activation='relu6', num_classes=10, width_multiplier=1.):
    return MobileNetV2(activation=activation,
                       num_classes=num_classes,
                       width_multiplier=width_multiplier)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    net = mobilenet_v2()
    macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
