#!/usr/bin/env python

from helpers.runner import runner
from helpers.loop import run_loop
from helpers.save_load import State
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from pathlib import Path

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

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 42, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(3, 42, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(3, 42, 3, 1),
            nn.LeakyReLU(),
            nn.Dropout(0.05),
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

        # x = self.conv1(x)
        # x = F.leaky_relu(x)
        # x = self.conv2(x)
        # x = F.leaky_relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


frame_number = 0


def dampImg(model: Net, index: int, name: str, dataset: datasets.ImageFolder, device):
    p = Path(f'tmp/facesAutoencoder/{index}')
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        img = dataset[index][0]
        save_image(img, f'tmp/facesAutoencoder/{index}/000.png')

    img = dataset[index][0].to(device).view(1, 3, 218, 178)
    out = model(img).to('cpu').view(3, 218, 178)
    save_image(out, f'tmp/facesAutoencoder/{index}/{name}_{frame_number}.png')
    # weight = model.conv1.cpu().weight
    # print(weight.data.shape)
    # data = weight.data[:, 0]
    # # data = weight.data[:, 0][2:]
    # # data = data.view(-1, 3, 3, 3)
    # print(data.shape)
    # # print(x[0])
    # for i, layer in enumerate(data):
    #     # print(i, layer.shape)
    #     save_image(layer, f'tmp/feature{i}.png')
    #     # print(i, model.conv1.weight.data[i:0].numpy())

def main():
    args, kwargs, device, = runner("FACES_conv_autoencoder", options)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('../data', train=False, download=False, transform=transform)
    # dataset_me = datasets.ImageFolder('../data/img_align_celeba/me', transform=transform)
    # dataset1 = datasets.CelebA('../data', split='train', download=True, transform=transform)
    # dataset2 = datasets.CelebA('../data', split='valid', transform=transform)
    # dataset_len = len(dataset)
    # print('==> len', len(dataset))
    # train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    # test_set, train_set = torch.utils.data.random_split(dataset, [1000, dataset_len-1000], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    state = State('save', 'FACES_conv_autoencoder', model, optimizer, args)
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

    # with torch.no_grad():
    #     x = torch.rand(1, 3, 218, 178)
    #     print('*' * 42)

    #     # x = calc_conv2d(x, 3, 3, 1)
    #     # x = calc_conv2d(x, 3, 4, 2)
    #     # x = calc_conv2d(x, 3, 5, 2)
    #     # x = calc_conv2d(x, 3, 4, 2)
    #     # x = calc_conv2d(x, 3, 3, 1)
    #     # x = calc_conv2d(x, 3, 3, 1)
    #     # x = calc_conv2d(x, 3, 3, 1)
    #     # x = calc_conv2d(x, 3, 3, 1)
    #     # x = calc_conv2d(x, 3, 3, 1)

    #     print('in --', x.data.shape)

    #     x = nn.Conv2d(3, 3, 3, 1)(x)
    #     x = nn.Conv2d(3, 3, 4, 2)(x)
    #     x = nn.Conv2d(3, 3, 5, 2)(x)
    #     x = nn.Conv2d(3, 3, 4, 2)(x)
    #     x = nn.Conv2d(3, 3, 3, 1)(x)
    #     x = nn.Conv2d(3, 3, 3, 1)(x)
    #     x = nn.Conv2d(3, 3, 3, 1)(x)
    #     x = nn.Conv2d(3, 3, 3, 1)(x)
    #     x = nn.Conv2d(3, 3, 3, 1)(x)

    #     print('---', x.data.shape)

    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
    #     x = nn.ConvTranspose2d(3, 3, 4, 2)(x)
    #     x = nn.ConvTranspose2d(3, 3, 5, 2)(x)
    #     x = nn.ConvTranspose2d(3, 3, 4, 2)(x)
    #     x = nn.ConvTranspose2d(3, 3, 3, 1)(x)

    #     print('out --', x.data.shape)
        # x = nn.Conv2d(3, 3, 3, 2)(x)
        # print('2--', x.data.shape)

        # calc_conv2d(x, 3, 4, 2)
        # x = nn.Conv2d(3, 3, 4, 2)(x)
        # # print('3--', x.data.shape)

        # calc_conv2d(x, 3, 5, 2)
        # x = nn.Conv2d(3, 3, 5, 2)(x)
        # # print('4--', x.data.shape)

        # calc_conv2d(x, 3, 3, 1)
        # x = nn.Conv2d(3, 3, 3, 1)(x)
        # print('---', x.data.shape)

        # x = nn.ConvTranspose2d(3, 3, 3, 1)(x)
        # print('4--', x.data.shape)

        # x = nn.ConvTranspose2d(3, 3, 5, 2)(x)
        # print('3--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 60, 5, 2)(x)
        # print('3--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 60, 5, 2)(x)
        # print('3--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 60, 4, 2)(x)
        # print('2--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 60, 3, 2)(x)
        # print('2--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 60, 3, 1)(x)
        # print('2--', x.data.shape)

        # x = nn.ConvTranspose2d(60, 42, 4, 2)(x)
        # print('4--', x.data.shape)
        # print('-'*5)
        # x = nn.Conv2d(40, 20, 3)(x)

        # print('--', x.data.shape)
        # x = nn.ConvTranspose2d(20, 40, 3)(x)
        # print('--', x.data.shape)
        # x = nn.ConvTranspose2d(40, 60, 5, 3)(x)
        # print('1--', x.data.shape)
        # x = nn.ConvTranspose2d(60, 42, 5, 3)(x)
        # x = nn.ConvTranspose2d(42, 22, 3, 3)(x)
        # x = nn.ConvTranspose2d(22, 3, 3, 2)(x)
        # print('out', x.data.shape)

        # print(x.data.shape)

    # p = dataset1.
    # print(p)

    # plt.imshow(x.permute(1, 2, 0))
    # plt.show()

    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 3, 1)
    # plt.imshow(dataset1[2][0].permute(1, 2, 0))
    # plt.subplot(1, 3, 2)
    # plt.imshow(dataset1[3][0].permute(1, 2, 0))
    # plt.subplot(1, 3, 3)
    # plt.imshow(dataset1[4][0].permute(1, 2, 0))

    # plt.show()

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
            scheduler.step()
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
    test(epoch)


if __name__ == '__main__':
    main()
