import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# PyTorch/XLA
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass

try:
    import torch_xla.debug.metrics as met
except ImportError:
    pass

try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pass

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ---------------------- Functions for GPU Training ---------------------- #


def evaluate_accuracy(loader, net, device=None):
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            if isinstance(net, nn.Module):
                net.eval()
                out = net(data.to(device))
                pred = out.argmax(dim=1, keepdim=True)
                acc_sum += pred.eq(
                    target.view_as(pred).to(device)).sum().item()
                net.train()
            else:
                raise NotImplementedError
            n += target.shape[0]
    return acc_sum / n


def train_model(net,
                train_loader,
                test_loader,
                batch_size,
                optimizer,
                scheduler,
                device,
                num_epochs,
                comment='DenseNet_C10'):
    writer = SummaryWriter(comment=comment)
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    print("training on ", device)

    for epoch in range(num_epochs):
        start = time.time()
        loss_sum, train_acc_sum, sample_count, batch_count = 0.0, 0.0, 0, 0

        for data, target in train_loader:
            # Fit NN model
            data = data.to(device)
            target = target.to(device)
            out = net(data)
            loss = loss_fn(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute the sum of loss and training accuracy
            loss_sum += loss.cpu().item()
            pred = out.argmax(dim=1, keepdim=True)
            train_acc_sum += pred.eq(
                target.view_as(pred).to(device)).sum().item()
            sample_count += target.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_loader, net)
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


def load_data_cifar_10(batch_size, resize=None, root='/tmp/cifar10'):
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

    train_loader = DataLoader(cifar10_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(cifar10_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    return train_loader, test_loader


# ---------------------- Functions for TPU Training ---------------------- #


def load_cifar_10_xla(batch_size, root='/tmp/cifar10'):
    """Download and load the CIFAR-10 dataset."""
    if not xm.is_master_ordinal():
        # Barrier: Wait until master is done downloading
        xm.rendezvous('download_only_once')

    # Get and shard dataset into dataloaders
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

    if xm.is_master_ordinal():
        # Barrier: Master done downloading, other workers can proceed
        xm.rendezvous('download_only_once')

    train_sampler = DistributedSampler(cifar10_train,
                                       num_replicas=xm.xrt_world_size(),
                                       rank=xm.get_ordinal(),
                                       shuffle=True)
    train_loader = DataLoader(cifar10_train,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=4,
                              drop_last=True)
    test_loader = DataLoader(cifar10_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True)

    return train_loader, test_loader


def test_loop_fn(loader, net):
    total_samples = 0
    correct = 0
    net.eval()
    data, pred, target = None, None, None
    for data, target in loader:
        out = net(data)
        pred = out.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += data.size()[0]

    accuracy = 100.0 * correct / total_samples
    print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy),
          flush=True)
    return accuracy, data, pred, target


def train_loop_fn(loader, net, optimizer, loss_fn, batch_size, log_steps):
    tracker = xm.RateTracker()
    net.train()
    for x, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, target)
        loss.backward()
        xm.optimizer_step(optimizer)
        tracker.add(batch_size)
        if x % log_steps == 0:
            print(
                '[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'
                .format(xm.get_ordinal(), x, loss.item(), tracker.rate(),
                        tracker.global_rate(), time.asctime()),
                flush=True)


def train_model_xla(net,
                    batch_size,
                    lr,
                    num_epochs,
                    log_steps=20,
                    metrics_debug=False):
    torch.manual_seed(1)

    train_loader, test_loader = load_cifar_10_xla(batch_size)

    # Scale learning rate to num cores
    lr = lr * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, num_epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device), net, optimizer,
                      loss_fn, batch_size, log_steps)
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        accuracy, data, pred, target = test_loop_fn(
            para_loader.per_device_loader(device), net)
        if metrics_debug:
            xm.master_print(met.metrics_report(), flush=True)

    return accuracy, data, pred, target
