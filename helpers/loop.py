from torch.utils.data import DataLoader
from torch import device

def run_loop(epoch: int, epochs: int, loader: DataLoader, device: device):
    for epoch in range(epoch, epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            yield epoch, batch_idx, data, target
