import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, game, numResBlocks, numHidden):
        super().__init__()
        self.startBlock = nn.Sequential(nn.Conv2d(13, numHidden, kernel_size=3, padding=1), nn.BatchNorm2d(numHidden), nn.ReLU())
        self.backBone = nn.ModuleList([ResBlock(numHidden) for i in range(numResBlocks)])
        self.policyHead = nn.Sequential(nn.Conv2d(numHidden, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Flatten(), nn.Linear(32*game.rowCount*game.colCount, game.actionSize))
        self.valueHead = nn.Sequential(nn.Conv2d(numHidden, 13, kernel_size=3, padding=1), nn.BatchNorm2d(13), nn.ReLU(), nn.Flatten(), nn.Linear(13*game.rowCount*game.colCount, 1), nn.Tanh())
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value 
        

class ResBlock(nn.Module):
    def __init__(self, numHidden):
        super().__init__()
        self.conv1 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(numHidden)
        self.conv2 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(numHidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x