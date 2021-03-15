import torch
import torch.nn as nn
import math
from pathlib import Path
import os
class State():
    def __init__(self, path: str, name: str, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.history_train = self._create_empty_buffer()
        self.history_test = self._create_empty_buffer_test()
        self.best_train_acc = 0
        self.best_train_loss = float('inf')
        self.epoch = 0
        self.path = path
        self.name = name

        self.path_last = f'{self.path}/{self.name}_last.nn'
        self.path_acc = f'{self.path}/{self.name}_best_acc.nn'
        self.path_loss = f'{self.path}/{self.name}_best_loss.nn'

        Path(path).mkdir(parents=True, exist_ok=True)

    def add_to_history(
        self,
        epoch: int,
        batch_id: int,
        batch_size: int,
        lr: float,
        loss: float,
        accuracy: float,
    ):
        self.history_train["epoch"].append(epoch)
        self.history_train["batch_id"].append(batch_id)
        self.history_train["batch_size"].append(batch_size)
        self.history_train["lr"].append(lr)
        self.history_train["loss"].append(loss)
        self.history_train["accuracy"].append(accuracy)

        self.epoch = epoch

        return self

    def add_to_test_history(
        self,
        epoch: int,
        lr: float,
        loss: float,
        accuracy: float,
    ):
        self.history_test["epoch"].append(epoch)
        self.history_test["lr"].append(lr)
        self.history_test["loss"].append(loss)
        self.history_test["accuracy"].append(accuracy)

        return self

    def save_net(self):
        last_loss = self.history_test['loss'][-1]
        last_acc = self.history_test['accuracy'][-1]
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "loss": last_loss,
            "acc": last_acc,
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(data, self.path_last)

        if self.best_train_acc <= last_acc:
            torch.save(data, self.path_acc)
            self.best_train_acc = last_acc

        if self.best_train_loss >= last_loss:
            torch.save(data, self.path_loss)
            self.best_train_loss = last_loss

        return self

    def load_net(self, choice: str):
        if not choice:
            return

        if not os.path.exists(self.path_last):
            raise Exception('cant load')

        data = None
        data_last = torch.load(self.path_last)
        data_acc = torch.load(self.path_acc)
        data_loss = torch.load(self.path_loss)
        self.best_train_loss = data_loss["loss"]
        self.best_train_acc = data_acc["acc"]

        if choice == 'acc':
            print('Load NN: acc')
            data = data_acc
        elif choice == 'loss':
            print('Load NN: loss')
            data = data_loss
        elif choice == 'last':
            print('Load NN: last')
            data = data_last

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])

        print(f'Loaded epoch: {self.epoch}, loss: {data["loss"]}, acc: {data["acc"]}')

        return self

    def log_last_train(self, batch_len: int, dataset_len: int, correct: int):
        batch_id = self.history_train["batch_id"][-1]
        epoch = self.history_train["epoch"][-1]
        loss = self.history_train["loss"][-1]
        acc = self.history_train["accuracy"][-1]
        passed = batch_id * batch_len
        progress = math.floor(passed/dataset_len * 100)

        print(f'Train Epoch: {epoch}\t [{passed}/{dataset_len} {progress}%]\t Loss: {loss:.6f} Acc: [{correct}/{batch_len} {acc:.3f}%]')

        return self

    def log_last_test(self, dataset_len: int, correct: int):
        epoch = self.history_test["epoch"][-1]
        loss = self.history_test["loss"][-1]
        acc = self.history_test["accuracy"][-1]
        lr = self.history_test["lr"][-1]

        print('-' * 42)
        print(f'Test Epoch: {epoch}\tLR: {lr}\tLoss: {loss:.6f}\tAcc: [{correct}/{dataset_len} {acc:.3f}%]')
        print('-' * 42)

        return self

    def _create_empty_buffer(self):
        return {
            "epoch": [],
            "batch_id": [],
            "batch_size": [],
            "lr": [],
            "loss": [],
            "accuracy": [],
        }

    def _create_empty_buffer_test(self):
        return {
            "epoch": [],
            "lr": [],
            "loss": [],
            "accuracy": [],
        }
