import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, game, numResBlocks, numHidden, device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(13, numHidden, kernel_size=3, padding=1), nn.BatchNorm2d(numHidden), nn.ReLU()
        ).to(f"{self.device}")
        self.backBone = nn.ModuleList([ResBlock(numHidden, device) for i in range(numResBlocks)]).to(f"{self.device}")
        self.policyHead = nn.Sequential(
            nn.Conv2d(numHidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rowCount * game.colCount, game.actionSize),
        ).to(f"{self.device}")
        self.valueHead = nn.Sequential(
            nn.Conv2d(numHidden, 13, kernel_size=3, padding=1),
            nn.BatchNorm2d(13),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(13 * game.rowCount * game.colCount, 1),
            nn.Tanh(),
        ).to(f"{self.device}")
        self.to(device)

    def forward(self, x):
        x = x.to(f"{self.device}")
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, numHidden, device):
        super().__init__()
        self.conv1 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1).to(f"{device}")
        self.bn1 = nn.BatchNorm2d(numHidden).to(f"{device}")
        self.conv2 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1).to(f"{device}")
        self.bn2 = nn.BatchNorm2d(numHidden).to(f"{device}")

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
