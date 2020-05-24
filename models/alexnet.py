import torch
import torch.nn as nn

from activations import Swish, Mish


class AlexNet(nn.Module):
    def __init__(self, activation='relu', num_classes=10):
        super(AlexNet, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'mish':
            self.activation = Mish()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            raise NotImplementedError

        # Convolutional part.
        # It's different from the original implementation cause the image size of CIFAR dataset is 32x32.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1), self.activation,
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(96, 256, kernel_size=5, stride=1,
                      padding=2), self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1,
                      padding=1), self.activation,
            nn.Conv2d(384, 384, kernel_size=3, stride=1,
                      padding=1), self.activation,
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            self.activation, nn.MaxPool2d(kernel_size=3, stride=2))
        # Fully connected part
        self.fc = nn.Sequential(nn.Linear(256 * 5 * 5, 4096), self.activation,
                                nn.Dropout(0.5), nn.Linear(4096, 4096),
                                self.activation, nn.Dropout(0.5),
                                nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    data = torch.zeros(1, 3, 32, 32)
    net = AlexNet(activation='mish')

    out = net(data)
    print(out.shape)
