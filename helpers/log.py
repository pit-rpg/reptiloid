def log(epoch, batch_idx, log_interval, loader, data, loss):
    if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(loader.dataset),
            100. * batch_idx / len(loader), loss.item()))
