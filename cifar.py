#!/usr/bin/env python

from helpers.runner import runner
from helpers.loop import run_loop
from helpers.save_load import State
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from pathlib import Path

options = {
    "--lr": 1.0,
    "--no-cuda": False,
    "--batch-size": 42,
    "--epochs": 21,
    "--log-interval": 10,
    "--seed": 42,
    "--gamma": 0.07,
}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, 42, 4, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(42, 100, 4, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(100, 300, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2d(300, 500, 3, 1),
            nn.Conv2d(500, 1000, 3, 1),
            nn.Flatten(),
        )

        self.tail = nn.Sequential(
            nn.Linear(1000, 10),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.tail(x)
        return x

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infographics(args, model: Net):
    if not args.infographics:
        return
    Path('tmp/').mkdir(parents=True, exist_ok=True)
    print('='*42)
    weight = model.conv1.cpu().weight
    print(weight.data.shape)
    data = weight.data[:, 0]
    # data = weight.data[:, 0][2:]
    # data = data.view(-1, 3, 3, 3)
    print(data.shape)
    # print(x[0])
    for i, layer in enumerate(data):
        # print(i, layer.shape)
        save_image(layer, f'tmp/feature{i}.png')
        # print(i, model.conv1.weight.data[i:0].numpy())


def main():
    args, kwargs, device, = runner("CIFAR Test", options)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10('../data', train=False, download=False, transform=transform)
    test_set = datasets.CIFAR10('../data', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

    model = Net().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    state = State('save', 'CIFAR_conv', model, optimizer, args)
    state.load_net()

    current_epoch = state.epoch

    infographics(args, model)

    # def calc_conv2d(tensor: torch.Tensor, inout: int, convs: int, stride: int):
    #     _, _, h1, w1 = tensor.data.shape
    #     h = (h1 - convs + 1) / stride
    #     w = (w1 - convs + 1) / stride
    #     tensor = nn.Conv2d(inout, inout, convs, stride)(tensor)
    #     tensor2 = nn.ConvTranspose2d(inout, inout, convs, stride)(tensor)
    #     _, _, h2, w2 = tensor2.data.shape
    #     print(f'{h}, {w} || {h1}, {w1} || {h2}, {w2} || {h1 == h2}, {w1 == w2}')
    #     return tensor

    # x = torch.rand(1, 3, 32, 32)

    # x = calc_conv2d(x, 3, 4, 1)
    # x = calc_conv2d(x, 3, 4, 2)
    # x = calc_conv2d(x, 3, 4, 2)
    # x = calc_conv2d(x, 3, 3, 1)
    # # x = calc_conv2d(x, 3, 4, 2)

    # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()
    # criterion = nn.MSELoss()

    if args.skip_train:
        return

    def train(train: bool, data, target, data_len):
        optimizer.zero_grad()
        model.train(train)
        output = None
        loss = None
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        if train:
            output = model(data)
            # print(output)
            # print(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
        with torch.no_grad():
            x1 = output.argmax(dim=1, keepdim=True)
            x2 = target.argmax(dim=1, keepdim=True)
            correct = x1.eq(x2).sum().item()
            acc = 100. * correct / data_len
            return loss.item(), acc, correct

    def test(epoch):
        if args.skip_train:
            return
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


if __name__ == '__main__':
    main()
