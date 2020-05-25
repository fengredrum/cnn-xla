import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Swish, Mish


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(ResidualBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.trunk(x)
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet(nn.Module):
    def __init__(self,
                 residual_block,
                 num_blocks,
                 activation='relu',
                 num_classes=10):
        super(ResNet, self).__init__()
        assert len(num_blocks) == 4, 'Invalid Conv Number!'

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

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
                                       stride=1)
        self.layer2 = self._make_layer(residual_block,
                                       128,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(residual_block,
                                       256,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(residual_block,
                                       512,
                                       num_blocks[3],
                                       stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.num_channels, out_channels, stride))
            self.num_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def resnet18(activation='relu', num_classes=10):
    return ResNet(ResidualBlock, [2, 2, 2, 2],
                  activation=activation,
                  num_classes=num_classes)


if __name__ == "__main__":
    x = torch.zeros(1, 3, 32, 32)
    net = resnet18(activation='mish')

    out = net(x)
    print(out.shape)
