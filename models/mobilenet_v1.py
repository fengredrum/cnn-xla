'''
A PyTorch implementation of MobileNetV1.
The original paper can be found at https://arxiv.org/abs/1704.04861.
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import activetion_func


# Depthwise Separable convolutions
class DepthSepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(DepthSepBlock, self).__init__()
        self.activation = activetion_func(activation)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=in_channels,
                      bias=False), nn.BatchNorm2d(in_channels),
            self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), self.activation)

    def forward(self, x):
        return self.net(x)


class MobileNetV1(nn.Module):
    def __init__(self,
                 activation='relu',
                 num_classes=10,
                 width_multiplier=1.0):
        super(MobileNetV1, self).__init__()
        self.activation = activetion_func(activation)
        self.out_channels = np.asarray([32, 64, 128, 256, 512, 1024])
        self.out_channels = np.floor(
            self.out_channels * width_multiplier).astype(np.int32).tolist()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3,
                      self.out_channels[0],
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(self.out_channels[0]),
            self.activation,
            DepthSepBlock(self.out_channels[0],
                          self.out_channels[1],
                          stride=1,
                          activation=activation))

        self.layer2 = self._make_layers(activation)
        self.linear = nn.Linear(self.out_channels[-1], num_classes)

    def _make_layers(self, activation):
        net = []
        for i in range(1, len(self.out_channels) - 1):
            if i == 3:
                num_blocks = 5
            else:
                num_blocks = 1

            layer = DepthSepBlock(self.out_channels[i],
                                  self.out_channels[i + 1],
                                  stride=2,
                                  activation=activation)
            net.append(layer)
            for _ in range(num_blocks):
                layer = DepthSepBlock(self.out_channels[i + 1],
                                      self.out_channels[i + 1],
                                      stride=1,
                                      activation=activation)
                net.append(layer)

        return nn.Sequential(*net)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def mobilenet_v1(activation='relu', num_classes=10, width_multiplier=1.0):
    return MobileNetV1(activation=activation,
                       num_classes=num_classes,
                       width_multiplier=width_multiplier)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    net = mobilenet_v1(activation='mish')
    macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
