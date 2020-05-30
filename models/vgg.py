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
        self.linear = nn.Sequential(
            nn.Linear(np.prod(feature_size), num_hiddens), self.activation,
            nn.Dropout(0.5), nn.Linear(num_hiddens, num_hiddens),
            self.activation, nn.Dropout(0.5), nn.Linear(num_hiddens, 10))

    def _get_feature_size(self, net):
        x = torch.zeros(1, 3, self.image_size, self.image_size)
        out = net(x)
        return out.size()[1:]

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        return self.linear(out)


def vgg11(image_size=32, ratio=8, activation='relu', num_classes=10):
    conv_arch = ((1, 3, 64 // ratio), (1, 64 // ratio, 128 // ratio),
                 (2, 128 // ratio, 256 // ratio),
                 (2, 256 // ratio, 512 // ratio), (2, 512 // ratio,
                                                   512 // ratio))
    num_hiddens = 4096 // ratio
    net = VGG(conv_arch,
              image_size,
              num_hiddens,
              activation=activation,
              num_classes=num_classes)
    return net


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    image_size = 32
    net = vgg11(activation='relu', ratio=1, image_size=image_size)
    macs, params = get_model_complexity_info(net, (3, image_size, image_size),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
