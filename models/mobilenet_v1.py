'''
A PyTorch implementation of MobileNetV1.
The original paper can be found at https://arxiv.org/abs/1704.04861.
'''

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
    def __init__(self, activation='relu', num_classes=10):
        super(MobileNetV1, self).__init__()
        self.activation = activetion_func(activation)
        num_channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,
                      num_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.BatchNorm2d(num_channels),
            self.activation,
            DepthSepBlock(num_channels,
                          num_channels * 2,
                          stride=1,
                          activation=activation))

        num_channels *= 2
        self.layer2, num_channels = self._make_layers(num_channels, activation)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_layers(self, num_channels, activation):
        net = []
        for i in range(8):
            if i == 5:
                num_blocks = 5
            else:
                num_blocks = 1

            stride = (i + 1) % 2 + 1
            for _ in range(num_blocks):
                layer = DepthSepBlock(num_channels, num_channels * stride,
                                      stride, activation)
                net.append(layer)
                num_channels *= stride

        return nn.Sequential(*net), num_channels

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 2)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def mobilenet_v1(activation='relu', num_classes=10):
    return MobileNetV1(activation=activation, num_classes=num_classes)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    net = mobilenet_v1(activation='mish')
    macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
