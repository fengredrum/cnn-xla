import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Swish, Mish


class DenseBlock(nn.Module):
    def __init__(self,
                 num_convs,
                 in_channels,
                 out_channels,
                 activation='relu'):
        super(DenseBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(self._block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels

    def _block(self, in_channels, out_channels):
        blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return blk

    def forward(self, x):
        for blk in self.net:
            out = blk(x)
            x = torch.cat((x, out), dim=1)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(TransitionBlock, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels), self.activation,
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self,
                 dense_block,
                 transit_block,
                 num_convs,
                 growth_rate=32,
                 activation='relu',
                 num_classes=10):
        super(DenseNet, self).__init__()
        assert len(num_convs) == 4, 'Invalid Conv Number!'

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        num_channels = 64
        self.conv1 = nn.Conv2d(3,
                               num_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.layer1, num_channels = self._make_layer(dense_block,
                                                     transit_block,
                                                     num_convs[0],
                                                     num_channels, growth_rate,
                                                     activation)
        self.layer2, num_channels = self._make_layer(dense_block,
                                                     transit_block,
                                                     num_convs[1],
                                                     num_channels, growth_rate,
                                                     activation)
        self.layer3, num_channels = self._make_layer(dense_block,
                                                     transit_block,
                                                     num_convs[2],
                                                     num_channels, growth_rate,
                                                     activation)
        self.layer4, num_channels = self._make_layer(dense_block,
                                                     transit_block,
                                                     num_convs[3],
                                                     num_channels,
                                                     growth_rate,
                                                     activation,
                                                     use_transit=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_layer(self,
                    dense_block,
                    transit_block,
                    num_convs,
                    num_channels,
                    growth_rate,
                    activation='relu',
                    use_transit=True):
        blk = dense_block(num_convs, num_channels, growth_rate, activation)
        num_channels = blk.out_channels
        layers = [blk]
        if use_transit:
            layers.append(
                transit_block(num_channels, num_channels // 2, activation))
            num_channels = num_channels // 2
        return nn.Sequential(*layers), num_channels

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.activation(self.bn2(out))
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        return self.linear(out)
        return out


def densenet18(activation='relu', num_classes=10):
    return DenseNet(DenseBlock,
                    TransitionBlock, [4, 4, 4, 4],
                    activation=activation,
                    num_classes=num_classes)


if __name__ == "__main__":
    data = torch.zeros(1, 3, 32, 32)
    net = densenet18(activation='mish')

    out = net(data)
    print(out.shape)
