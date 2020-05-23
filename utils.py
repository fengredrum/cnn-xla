import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                output = net(X.to(device))
                pred = output.argmax(dim=1, keepdim=True)
                acc_sum += pred.eq(y.view_as(pred).to(device)).sum().item()
                net.train()
            else:
                raise NotImplementedError
            n += y.shape[0]
    return acc_sum / n


def train_model(net,
                train_iter,
                test_iter,
                batch_size,
                optimizer,
                scheduler,
                device,
                num_epochs,
                comment='DenseNet_C10'):
    writer = SummaryWriter(comment=comment)
    net = net.to(device)
    loss_func = nn.CrossEntropyLoss()
    print("training on ", device)

    for epoch in range(num_epochs):
        start = time.time()
        loss_sum, train_acc_sum, sample_count, batch_count = 0.0, 0.0, 0, 0

        for x, y in train_iter:
            # Fit NN model
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute the sum of loss and training accuracy
            loss_sum += loss.cpu().item()
            pred = out.argmax(dim=1, keepdim=True)
            train_acc_sum += pred.eq(y.view_as(pred).to(device)).sum().item()
            sample_count += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        scheduler.step()
        # Training status
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, loss_sum / batch_count, train_acc_sum / sample_count,
               test_acc, time.time() - start))
        # Log stuffs
        writer.add_scalar('loss', loss_sum / batch_count, epoch + 1)
        writer.add_scalar('train acc', train_acc_sum / sample_count, epoch + 1)
        writer.add_scalar('test acc', test_acc, epoch + 1)


def load_data_cifar_10(batch_size, resize=None, root='/tmp/CIFAR10'):
    """Download and load the CIFAR-10 dataset."""
    norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                std=(0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), norm])

    cifar10_train = datasets.CIFAR10(root=root,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    cifar10_test = datasets.CIFAR10(root=root,
                                    train=False,
                                    download=True,
                                    transform=transform_test)

    train_iter = DataLoader(cifar10_train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    test_iter = DataLoader(cifar10_test,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4)

    return train_iter, test_iter
