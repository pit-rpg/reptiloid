import torch
import torch.nn as nn
import statistics
import math

class Store():
    def __init__(self, path: str, name: str, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.buffer = self._create_empty_buffer()
        self.history_train = self._create_empty_buffer()
        self.history_test = self._create_empty_buffer_test()
        self.best_train_acc = 0
        self.best_train_loss = float('inf')
        self.epoch = 0

    def buffer_add(
        self,
        epoch: int,
        batch_id: int,
        batch_size: int,
        lr: float,
        loss: float,
        accuracy: float,
    ):
        self.buffer["epoch"].append(epoch)
        self.buffer["batch_id"].append(batch_id)
        self.buffer["batch_size"].append(batch_size)
        self.buffer["lr"].append(lr)
        self.buffer["loss"].append(loss)
        self.buffer["accuracy"].append(accuracy)

        self.epoch = epoch

        return self

    def buffer_to_train_history(self):
        self.history_train["epoch"] += self.buffer["epoch"]
        self.history_train["batch_id"] += self.buffer["batch_id"]
        self.history_train["batch_size"] += self.buffer["batch_size"]
        self.history_train["lr"] += self.buffer["lr"]
        self.history_train["loss"] += self.buffer["loss"]
        self.history_train["accuracy"] += self.buffer["accuracy"]

        self.buffer = self._create_empty_buffer()

        return self

    def buffer_to_test_history(self):
        self.history_train["epoch"] += self.epoch
        self.history_train["lr"] += statistics.mean(self.buffer["lr"])
        self.history_train["loss"] += statistics.mean(self.buffer["loss"])
        self.history_train["accuracy"] += statistics.mean(self.buffer["accuracy"])

        self.buffer = self._create_empty_buffer()

        return self

    def save_net(self):
        last_loss = self.history_train['loss'][-1]
        last_acc = self.history_train['accuracy'][-1]
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(f'{self.path}/{self.name}_last.nn', data)

        if self.best_train_acc <= last_acc:
            torch.save(f'{self.path}/{self.name}_best_acc.nn', data)
            self.best_train_acc = last_acc

        if self.best_train_loss <= last_loss:
            torch.save(f'{self.path}/{self.name}_best_loss.nn', data)
            self.best_train_loss = last_loss

        return self

    def load_net(self, path: str):
        data = torch.load(path)

        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])

        return self

    def log_last_train(self, batch_len: int, dataset_len: int):
        batch_id = self.buffer["batch_id"][-1]
        epoch = self.buffer["epoch"][-1]
        loss = self.buffer["loss"][-1]
        acc = self.buffer["accuracy"][-1]
        passed = batch_id * batch_len
        progress = math.floor(passed/dataset_len * 100)

        print(f'Train Epoch: {epoch}\t [{passed}/{dataset_len}\t {progress}%]\t Loss: {loss} Acc: {acc}')

        return self

    def log_last_test(self):
        epoch = self.buffer["epoch"][-1]
        loss = self.buffer["loss"][-1]
        accuracy = self.buffer["accuracy"][-1]
        lr = self.buffer["lr"][-1]

        print(f'Test: {epoch}\t LR: {lr} Loss: {loss} Acc: {accuracy}')

        return self

    def _create_empty_buffer():
        return {
            "epoch": [int],
            "batch_id": [int],
            "batch_size": [int],
            "lr": [float],
            "loss": [float],
            "accuracy": [float],
        }

    def _create_empty_buffer_test():
        return {
            "epoch": [int],
            "lr": [float],
            "loss": [float],
            "accuracy": [float],
        }
