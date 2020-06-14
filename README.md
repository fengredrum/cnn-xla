# CNN-XLA

![License Badge](https://img.shields.io/badge/python-3.5%2B-blue) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-TPU.ipynb)

A collection of CNN models are trained on Cloud TPU by using PyTorch/XLA

# Get Started

* [Train on GPU](notebooks/Train-on-GPU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-GPU.ipynb)
* [Train on TPU](notebooks/Train-on-TPU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fengredrum/cnn-xla/blob/master/notebooks/Train-on-TPU.ipynb)

# CNN Models

| Model              | Input Resolution   | Params(M)          | MACs(G)            | Percentage Correct |
| ------------------ | :----------------: | :----------------: | :----------------: | :----------------: |
| [AlexNet](models/alexnet.py)   | 32x32 | 46.76 | 0.91 | - |
| [VGG-11](models/vgg.py)        | 32x32 | 28.14 | 0.17 | - |
| Inception                      | 32x32 | -     | -    | - |
| [ResNet-18](models/resnet.py)  | 32x32 | 11.17 | 0.56 | - |
| [DenseNet-121](models/densenet.py) | 32x32 | 6.69 | 0.9 | - |

# Related Repositories

* [Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
* [PyTorch/XLA](https://github.com/pytorch/xla)

# License

[MIT License](LICENSE)
