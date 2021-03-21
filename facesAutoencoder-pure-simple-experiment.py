#!/usr/bin/env python

from helpers.models import CIFARFeature
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
            nn.Conv2d(3, 42, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(42, 60, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2d(60, 70, 5, 2),
            nn.LeakyReLU(),
            nn.Conv2d(70, 80, 4, 2),
            nn.LeakyReLU(),
            nn.Conv2d(80, 90, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(90, 100, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(100, 120, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(120, 130, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(130, 140, 3, 1),
            nn.LeakyReLU(),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(140, 130, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(130, 120, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(120, 110, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(110, 100, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(100, 90, 3, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(90, 80, 4, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(80, 70, 5, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(70, 60, 4, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(60, 3, 3, 1),
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

def dampImg(model: Net, name: str, dataset: datasets.ImageFolder, device, alpha):
    img1 = dataset[0][0].to(device).view(1, 3, 218, 178)
    img2 = dataset[1][0].to(device).view(1, 3, 218, 178)

    model.train(False)

    img1 = inv_normalize(img1)
    # img1 = model(img1).view(3, 218, 178)
    img2 = model(img2).view(3, 218, 178)
    img1 = model(img1).view(3, 218, 178).cpu()

    # img = torch.lerp(img1, img2, 1.0)

    # print(img1)
    # print(img1.data.view)

    # out = inv_normalize(img1).cpu()
    # out = model(img)
    save_image(out, f'tmp/facesAutoencoder-pure-simple-experiment/{name}_{frame_number}.png')

def main():
    args, kwargs, device, = runner("FACES-pure-simple-experiment", options)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder('../data/img_align_celeba/img_align_celeba', transform=transform)
    dataset_me = datasets.ImageFolder('../data/img_align_celeba/me', transform=transform)
    dataset_len = len(dataset)
    test_set, train_set = torch.utils.data.random_split(dataset, [1000, dataset_len-1000], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state = State('save', 'FACES_conv_autoencoder-pure-simple-experiment', model, optimizer, args)
    state.load_net()

    current_epoch = state.epoch

    # dampImg(model, 5, 'text', train_set, device)
    dampImg(model, 'test', dataset_me, device, 0.5)

    # def calc_conv2d(tensor: torch.Tensor, inout: int, convs: int, stride: int):
    #     _, _, h1, w1 = tensor.data.shape
    #     h = (h1 - convs + 1) / stride
    #     w = (w1 - convs + 1) / stride
    #     tensor = nn.Conv2d(inout, inout, convs, stride)(tensor)
    #     tensor2 = nn.ConvTranspose2d(inout, inout, convs, stride)(tensor)
    #     _, _, h2, w2 = tensor2.data.shape
    #     print(f'{h}, {w} || {h1}, {w1} || {h2}, {w2} || {h1 == h2}, {w1 == w2}')
    #     return tensor

    # if args.skip_train:
    #     return

    # def criterion(output, data):
    #     loss1 = F.l1_loss(output, data)
    #     output = F.avg_pool2d(output, (2, 2))
    #     data = F.avg_pool2d(data, (2, 2))
    #     loss2 = F.l1_loss(output, data)
    #     return loss1 + loss2

    # def train(train: bool, data, target, data_len):
    #     optimizer.zero_grad()
    #     model.train(train)
    #     output = None
    #     loss = None
    #     if train:
    #         output = model(data)
    #         loss = criterion(output, data)
    #         loss.backward()
    #         optimizer.step()
    #     else:
    #         with torch.no_grad():
    #             output = model(data)
    #             loss = criterion(output, data)
    #     with torch.no_grad():
    #         # pred = output.argmax(dim=1, keepdim=True)
    #         # correct = pred.eq(target.view_as(pred)).sum().item()
    #         # acc = 100. * correct / data_len
    #         return loss.item(), loss.item(), loss.item()

    # def test(epoch):
    #     if args.skip_train:
    #         return
    #     loss, _, correct = 0, 0, 0
    #     data_len = len(test_loader.dataset)
    #     lr = get_lr(optimizer)

    #     for _, batch_idx, data, target in run_loop(0, 1, test_loader, device):
    #         _loss, _, _ = train(False, data, target, data_len)
    #         loss += _loss
    #     loss /= data_len
    #     state.add_to_test_history(epoch, lr, loss, loss)
    #     state.log_last_test(data_len, correct)
    #     state.save_net()

    # for epoch, batch_idx, data, target in run_loop(current_epoch, args.epochs, train_loader, device):
    #     if epoch != current_epoch and batch_idx == 0:
    #         test(epoch - 1)

    #     data_len = len(data)
    #     lr = get_lr(optimizer)

    #     loss, acc, correct = train(True, data, target, data_len)

    #     state.add_to_history(epoch, batch_idx, data_len, lr, loss, acc)

    #     del data

    #     global frame_number

    #     if batch_idx % args.log_interval == 0:
    #         state.log_last_train(data_len, len(train_loader.dataset), correct)
    #         frame_number += 1
    #         with torch.no_grad():
    #             dampImg(model, 5, f'{epoch}_{batch_idx}', train_set, device)
    #             dampImg(model, 0, f'{epoch}_{batch_idx}', dataset_me, device)

    #     if batch_idx % (args.log_interval * 20) == 0:
    #         state.save_last()

    #     signal.signal(signal.SIGINT, lambda x, y: state.save_last(True))
    #     signal.signal(signal.SIGTERM, lambda x, y: state.save_last(True))

    # test(epoch)


if __name__ == '__main__':
    main()