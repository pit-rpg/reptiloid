#!/usr/bin/env python

from helpers.runner import runner
from helpers.loop import run_loop
from helpers.save_load import State
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

options = {
    "--lr": 1.0,
    "--no-cuda": False,
    "--batch-size": 300,
    "--epochs": 21,
    "--log-interval": 10,
    "--seed": 42,
    "--gamma": 0.07,
    "--data": None,
    "--load-nn": None,
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    args, kwargs, device, = runner("MNIST Test", options)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    state = State('save', 'MNIST_conv', model, optimizer)

    state.load_net(args.load_nn)
    current_epoch = state.epoch

    def train(train: bool, data, target, data_len):
        optimizer.zero_grad()
        model.train(train)
        output = None
        loss = None
        if train:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data)
                loss = F.nll_loss(output, target)
        with torch.no_grad():
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / data_len
            return loss.item(), acc, correct

    def test(epoch):
        loss, _, correct = 0, 0, 0
        data_len = len(test_loader.dataset)
        lr = get_lr(optimizer)

        for _, batch_idx, data, target in run_loop(0, 1, test_loader, device):
            _loss, _acc, _correct = train(False, data, target, data_len)
            loss += _loss
            correct += _correct

        state.add_to_test_history(epoch, lr, loss/data_len, correct/data_len*100.)
        state.log_last_test(data_len, correct)
        state.save_net()

    for epoch, batch_idx, data, target in run_loop(current_epoch, args.epochs, train_loader, device):
        if epoch != current_epoch and batch_idx == 0:
            scheduler.step()
            test(epoch - 1)

        data_len = len(data)
        lr = get_lr(optimizer)

        loss, acc, correct = train(True, data, target, data_len)

        state.add_to_history(epoch, batch_idx, data_len, lr, loss, acc)

        if batch_idx % args.log_interval == 0:
            state.log_last_train(data_len, len(train_loader.dataset), correct)
    test(epoch)

    #     model.train(True)
    #     output = model(data)
    #     optimizer.zero_grad()
    #     loss = F.nll_loss(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     log(epoch, batch_idx, args.log_interval, train_loader, data, loss)
    #     if epoch != 0 and batch_idx == 0:

    #         scheduler.step()

    #         for param_group in optimizer.param_groups:
    #             print(param_group["lr"])
    #         print('-' * 42)
    #         print(model.conv1.weight.grad.mean())
    #         print(model.conv2.weight.grad.mean())
    #         print(model.fc1.weight.grad.mean())
    #         print(model.fc2.weight.grad.mean())

    #         print('-' * 42)

    #         print(model.conv1.weight.data.mean())
    #         print(model.conv2.weight.data.mean())
    #         print(model.fc1.weight.data.mean())
    #         print(model.fc2.weight.data.mean())

    #         model.train(False)
    #         test_loss = 0
    #         correct = 0
    #         for epoch, batch_idx, data, target in run_loop(1, test_loader, device):
    #             output = model(data)
    #             # sum up batch loss
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()
    #             # get the index of the max log-probability
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #         test_loss /= len(test_loader.dataset)

    #         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #             test_loss, correct, len(test_loader.dataset),
    #             100. * correct / len(test_loader.dataset)))
    # model.train(False)
    # test_loss = 0
    # correct = 0
    # for epoch, batch_idx, data, target in run_loop(1, test_loader, device):
    #     output = model(data)
    #     # sum up batch loss
    #     test_loss += F.nll_loss(output, target, reduction='sum').item()
    #     # get the index of the max log-probability
    #     pred = output.argmax(dim=1, keepdim=True)
    #     correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
