import torch
import torch.nn as nn
import torch.nn.functional as F


def activetion_func(activation_name):
    if activation_name == 'relu':
        activation = nn.ReLU()
    elif activation_name == 'relu6':
        activation = nn.ReLU6()
    elif activation_name == 'tanh':
        activation = nn.Tanh()
    elif activation_name == 'mish':
        activation = Mish()
    elif activation_name == 'swish':
        activation = Swish()
    else:
        raise NotImplementedError
    return activation


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x
