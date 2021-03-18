#!/usr/bin/env python

from helpers.runner import runner
from helpers.loop import run_loop
from helpers.save_load import State
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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

        self.encode = nn.Sequential(
            nn.Conv2d(3, 22, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(22, 30, 4, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(30, 30, 5, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(30, 32, 4, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 34, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(34, 36, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.02),
            nn.Conv2d(36, 24, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 16, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 4, 3, 1),
            nn.LeakyReLU(),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 24, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 24, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 36, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.ConvTranspose2d(36, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.ConvTranspose2d(32, 30, 4, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(30, 30, 5, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(30, 22, 4, 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(22, 3, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


frame_number = 0

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


def dampImg(model: Net, index: int, name: str, dataset: datasets.ImageFolder, device):
    p = Path(f'tmp/facesAutoencoder-main/{index}')
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        img = dataset[index][0]
        img = inv_normalize(img)
        save_image(img, f'tmp/facesAutoencoder-main/{index}/000.png')

    img = dataset[index][0].to(device).view(1, 3, 218, 178)
    img = inv_normalize(img)
    out = model(img).to('cpu').view(3, 218, 178)
    save_image(out, f'tmp/facesAutoencoder-main/{index}/{name}_{frame_number}.png')

def main():
    args, kwargs, device, = runner("FACES_conv_autoencoder-main", options)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder('../data/img_align_celeba/img_align_celeba', transform=transform)
    dataset_me = datasets.ImageFolder('../data/img_align_celeba/me', transform=transform)
    dataset_len = len(dataset)
    test_set, train_set = torch.utils.data.random_split(dataset, [1000, dataset_len-1000], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state = State('save', 'FACES_conv_autoencoder-main', model, optimizer, args)
    state.load_net()

    current_epoch = state.epoch

    def calc_conv2d(tensor: torch.Tensor, inout: int, convs: int, stride: int):
        _, _, h1, w1 = tensor.data.shape
        h = (h1 - convs + 1) / stride
        w = (w1 - convs + 1) / stride
        tensor = nn.Conv2d(inout, inout, convs, stride)(tensor)
        tensor2 = nn.ConvTranspose2d(inout, inout, convs, stride)(tensor)
        _, _, h2, w2 = tensor2.data.shape
        print(f'{h}, {w} || {h1}, {w1} || {h2}, {w2} || {h1 == h2}, {w1 == w2}')
        return tensor

    if args.skip_train:
        return

    criterion = nn.MSELoss()

    def train(train: bool, data, target, data_len):
        optimizer.zero_grad()
        model.train(train)
        output = None
        loss = None
        if train:
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, data)
        with torch.no_grad():
            # pred = output.argmax(dim=1, keepdim=True)
            # correct = pred.eq(target.view_as(pred)).sum().item()
            # acc = 100. * correct / data_len
            return loss.item(), loss.item(), loss.item()

    def test(epoch):
        if args.skip_train:
            return
        loss, _, correct = 0, 0, 0
        data_len = len(test_loader.dataset)
        lr = get_lr(optimizer)

        for _, batch_idx, data, target in run_loop(0, 1, test_loader, device):
            _loss, _, _ = train(False, data, target, data_len)
            loss += _loss
            # correct += _correct
        loss /= data_len
        state.add_to_test_history(epoch, lr, loss, loss)
        state.log_last_test(data_len, correct)
        state.save_net()

    for epoch, batch_idx, data, target in run_loop(current_epoch, args.epochs, train_loader, device):
        if epoch != current_epoch and batch_idx == 0:
            test(epoch - 1)

        data_len = len(data)
        lr = get_lr(optimizer)

        loss, acc, correct = train(True, data, target, data_len)

        state.add_to_history(epoch, batch_idx, data_len, lr, loss, acc)

        del data

        if batch_idx % args.log_interval == 0:
            state.log_last_train(data_len, len(train_loader.dataset), correct)
            global frame_number
            frame_number += 1
            dampImg(model, 5, f'{epoch}_{batch_idx}', train_set, device)
            dampImg(model, 0, f'{epoch}_{batch_idx}', dataset_me, device)

        if batch_idx % (args.log_interval * 20) == 0:
            state.save_last()

        signal.signal(signal.SIGINT, lambda x, y: state.save_last(True))
        signal.signal(signal.SIGTERM, lambda x, y: state.save_last(True))

    test(epoch)


if __name__ == '__main__':
    main()
