'''
A PyTorch implementation of DenseNet.
The original paper can be found at https://arxiv.org/abs/1608.06993.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import activetion_func


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(BottleneckBlock, self).__init__()
        self.activation = activetion_func(activation)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels), self.activation,
            nn.Conv2d(in_channels, 4 * out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * out_channels), self.activation,
            nn.Conv2d(4 * out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False))

    def forward(self, x):
        out = self.net(x)
        return torch.cat([x, out], dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(TransitionBlock, self).__init__()

        self.activation = activetion_func(activation)

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels), self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self,
                 num_blocks,
                 growth_rate=12,
                 reduction=0.5,
                 activation='relu',
                 num_classes=10):
        super(DenseNet, self).__init__()
        assert len(num_blocks) == 4, 'Invalid Conv Number!'
        self.activation = activetion_func(activation)

        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3,
                               num_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.layer1, num_channels = self._make_layer(BottleneckBlock,
                                                     TransitionBlock,
                                                     num_blocks[0],
                                                     num_channels, growth_rate,
                                                     reduction, activation)
        self.layer2, num_channels = self._make_layer(BottleneckBlock,
                                                     TransitionBlock,
                                                     num_blocks[1],
                                                     num_channels, growth_rate,
                                                     reduction, activation)
        self.layer3, num_channels = self._make_layer(BottleneckBlock,
                                                     TransitionBlock,
                                                     num_blocks[2],
                                                     num_channels, growth_rate,
                                                     reduction, activation)
        self.layer4, num_channels = self._make_layer(BottleneckBlock,
                                                     TransitionBlock,
                                                     num_blocks[3],
                                                     num_channels,
                                                     growth_rate,
                                                     reduction,
                                                     activation,
                                                     use_transit=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_layer(self,
                    dense_block,
                    transit_block,
                    num_blocks,
                    num_channels,
                    growth_rate,
                    reduction,
                    activation='relu',
                    use_transit=True):
        layers = []
        for _ in range(num_blocks):
            layers.append(dense_block(num_channels, growth_rate, activation))
            num_channels += growth_rate

        if use_transit:
            out_channels = int(math.floor(num_channels * reduction))
            layers.append(transit_block(num_channels, out_channels,
                                        activation))
            num_channels = out_channels
        return nn.Sequential(*layers), num_channels

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.activation(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        return self.linear(out)


def densenet121(activation='relu', num_classes=10):
    return DenseNet(num_blocks=[6, 12, 24, 16],
                    growth_rate=32,
                    activation=activation,
                    num_classes=num_classes)


def densenet169(activation='relu', num_classes=10):
    return DenseNet(num_blocks=[6, 12, 32, 32],
                    growth_rate=32,
                    activation=activation,
                    num_classes=num_classes)


def densenet201(activation='relu', num_classes=10):
    return DenseNet(num_blocks=[6, 12, 48, 32],
                    growth_rate=32,
                    activation=activation,
                    num_classes=num_classes)


def densenet264(activation='relu', num_classes=10):
    return DenseNet(num_blocks=[6, 12, 64, 48],
                    growth_rate=32,
                    activation=activation,
                    num_classes=num_classes)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    net = densenet121(activation='mish')
    macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
