import torch.nn as nn

class CIFARFeature(nn.Module):
    def __init__(self):
        super(CIFARFeature, self).__init__()

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
