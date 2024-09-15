from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import width
from utils import height


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3,  padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3,  padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3,  padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2))

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,  padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2))
        
        self.flatten = nn.Flatten()

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=30721, out_features=1024))

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128))

        self.fc_3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=3))

    def forward(self, x: torch.Tensor, speed: torch.Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.fc_1(torch.cat((x, speed), dim=1))
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    BATCH_SIZE = 5
    summary(net, input_data=(torch.randn(BATCH_SIZE, 3, width, height).to(device), torch.randn(BATCH_SIZE, 1).to(device)))

    data = []
    for i in range(100):
        data.append(torch.rand(BATCH_SIZE, 3, width, height).to(device))

    import time

    # start = time.time()
    # for i in range(1000):
    #     net(data[i], torch.rand(30,1).to(device))
    # end = time.time()
    # print(end - start)

