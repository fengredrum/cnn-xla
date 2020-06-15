'''
A PyTorch implementation of SE-ResNet-50.
The original paper can be found at https://arxiv.org/abs/1709.01507.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import activetion_func


class SEBlock(nn.Module):
    def __init__(self, spatial_dim, num_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=spatial_dim),
            nn.Conv2d(num_channels,
                      num_channels // reduction_ratio,
                      kernel_size=1), nn.ReLU(),
            nn.Conv2d(num_channels // reduction_ratio,
                      num_channels,
                      kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        out = self.net(x)
        out = out * x
        return out


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 spatial_dim,
                 stride=1,
                 expansion=4,
                 activation='relu'):
        super(BottleneckBlock, self).__init__()
        self.activation = activetion_func(activation)

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels), self.activation,
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=stride,
                      bias=False), nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels,
                      expansion * out_channels,
                      kernel_size=1,
                      bias=False), nn.BatchNorm2d(expansion * out_channels))

        if stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          expansion * out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(expansion * out_channels))
        else:
            self.shortcut = nn.Sequential()

        self.sqz_xit = SEBlock(spatial_dim, expansion * out_channels)

    def forward(self, x):
        out = self.trunk(x)
        out = self.sqz_xit(out)
        out += self.shortcut(x)
        return self.activation(out)


class SEResNet(nn.Module):
    def __init__(self,
                 residual_block,
                 num_blocks,
                 image_size=32,
                 expansion=4,
                 activation='relu',
                 num_classes=10):
        super(SEResNet, self).__init__()
        assert len(num_blocks) == 4, 'Invalid Conv Number!'

        self.image_size = image_size
        self.expansion = expansion
        self.activation = activetion_func(activation)

        self.num_channels = 64
        self.conv1 = nn.Conv2d(3,
                               self.num_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.layer1 = self._make_layer(residual_block,
                                       self.num_channels,
                                       num_blocks[0],
                                       stride=1,
                                       activation=activation)
        self.image_size = self.image_size // 2
        self.layer2 = self._make_layer(residual_block,
                                       128,
                                       num_blocks[1],
                                       stride=2,
                                       activation=activation)
        self.image_size = self.image_size // 2
        self.layer3 = self._make_layer(residual_block,
                                       256,
                                       num_blocks[2],
                                       stride=2,
                                       activation=activation)
        self.image_size = self.image_size // 2
        self.layer4 = self._make_layer(residual_block,
                                       512,
                                       num_blocks[3],
                                       stride=2,
                                       activation=activation)
        self.linear = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, activation):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.num_channels, out_channels, self.image_size, stride,
                      self.expansion, activation))
            self.num_channels = out_channels * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def se_resnet_50(activation='relu', num_classes=10):
    return SEResNet(BottleneckBlock, [3, 4, 6, 3],
                    image_size=32,
                    expansion=4,
                    activation=activation,
                    num_classes=num_classes)


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    net = se_resnet_50(activation='mish')
    macs, params = get_model_complexity_info(net, (3, 32, 32),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
