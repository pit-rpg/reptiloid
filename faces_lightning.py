#!/usr/bin/env python

from helpers.runner import runner, getArgs
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
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


class FacesNet(nn.Module):
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

class LitFaces(pl.LightningModule):

    def __init__(self, data_dir='../data', seed=42, batch_size=32):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.seed = seed
        # self.learning_rate = learning_rate

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )

        self.model = FacesNet()
        self.loader_options = {
            'batch_size': batch_size,
            'shuffle': True,
            # 'num_workers': multiprocessing.cpu_count(),
            # 'pin_memory': True,
        }

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x = batch
        x_heat = self(x)
        loss = F.l1_loss(x_heat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        if self.dataset:
            return

        self.dataset = datasets.ImageFolder('../data/img_align_celeba/img_align_celeba', transform=self.transform)
        count = len(self.dataset)
        generator = torch.Generator().manual_seed(self.seed)

        self.set_train, rest = torch.utils.data.random_split(self.dataset, [0, count - 2000], generator=generator)
        self.set_val, self.set_test = torch.utils.data.random_split(rest, [0, 1000], generator=generator)

    def train_dataloader(self):
        torch.utils.data.DataLoader(self.set_train, **self.loader_options)

    def val_dataloader(self):
        torch.utils.data.DataLoader(self.set_val, **self.loader_options)

    def test_dataloader(self):
        torch.utils.data.DataLoader(self.set_test, **self.loader_options)


def main():
    args = getArgs()


if __name__ == '__main__':
    main()


# options = {
#     "--lr": 1.0,
#     "--no-cuda": False,
#     "--batch-size": 42,
#     "--epochs": 21,
#     "--log-interval": 10,
#     "--seed": 42,
#     "--gamma": 0.07,
# }


# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.conv1 = nn.Conv2d(1, 32, 3, 1)
# #         self.conv2 = nn.Conv2d(32, 64, 3, 1)
# #         self.dropout1 = nn.Dropout(0.25)
# #         self.dropout2 = nn.Dropout(0.5)
# #         self.fc1 = nn.Linear(9216, 128)
# #         self.fc2 = nn.Linear(128, 10)

# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = F.leaky_relu(x)
# #         x = self.conv2(x)
# #         x = F.leaky_relu(x)
# #         x = F.max_pool2d(x, 2)
# #         x = self.dropout1(x)
# #         x = torch.flatten(x, 1)
# #         x = self.fc1(x)
# #         x = F.leaky_relu(x)
# #         x = self.dropout2(x)
# #         x = self.fc2(x)
# #         output = F.log_softmax(x, dim=1)
# #         return output

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

# def infographics(args, model: Net):
#     if not args.infographics:
#         return
#     Path('tmp/').mkdir(parents=True, exist_ok=True)
#     print('='*42)
#     weight = model.conv1.cpu().weight
#     print(weight.data.shape)
#     data = weight.data[:, 0]
#     # data = weight.data[:, 0][2:]
#     # data = data.view(-1, 3, 3, 3)
#     print(data.shape)
#     # print(x[0])
#     for i, layer in enumerate(data):
#         # print(i, layer.shape)
#         save_image(layer, f'tmp/feature{i}.png')
#         # print(i, model.conv1.weight.data[i:0].numpy())


# def main():
#     args, kwargs, device, = runner("MNIST Test", options)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize((0.1307,), (0.3081,))
#     ])

#     dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
#     dataset2 = datasets.MNIST('../data', train=False, transform=transform)

#     train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

#     model = Net().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr)
#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

#     state = State('save', 'MNIST_conv', model, optimizer, args)
#     state.load_net()

#     current_epoch = state.epoch

#     infographics(args, model)

#     if args.skip_train:
#         return

#     def train(train: bool, data, target, data_len):
#         optimizer.zero_grad()
#         model.train(train)
#         output = None
#         loss = None
#         if train:
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#         else:
#             with torch.no_grad():
#                 output = model(data)
#                 loss = F.nll_loss(output, target)
#         with torch.no_grad():
#             pred = output.argmax(dim=1, keepdim=True)
#             correct = pred.eq(target.view_as(pred)).sum().item()
#             acc = 100. * correct / data_len
#             return loss.item(), acc, correct

#     def test(epoch):
#         if args.skip_train:
#             return
#         loss, _, correct = 0, 0, 0
#         data_len = len(test_loader.dataset)
#         lr = get_lr(optimizer)

#         for _, batch_idx, data, target in run_loop(0, 1, test_loader, device):
#             _loss, _acc, _correct = train(False, data, target, data_len)
#             loss += _loss
#             correct += _correct

#         state.add_to_test_history(epoch, lr, loss/data_len, correct/data_len*100.)
#         state.log_last_test(data_len, correct)
#         state.save_net()

#     for epoch, batch_idx, data, target in run_loop(current_epoch, args.epochs, train_loader, device):
#         if epoch != current_epoch and batch_idx == 0:
#             scheduler.step()
#             test(epoch - 1)

#         data_len = len(data)
#         lr = get_lr(optimizer)

#         loss, acc, correct = train(True, data, target, data_len)

#         state.add_to_history(epoch, batch_idx, data_len, lr, loss, acc)

#         if batch_idx % args.log_interval == 0:
#             state.log_last_train(data_len, len(train_loader.dataset), correct)
#     test(epoch)


# if __name__ == '__main__':
#     main()
