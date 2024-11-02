import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 512),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x