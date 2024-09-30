from torchinfo import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

width = 640
# height = utils.height
height = 160


class Net(nn.Module):
    def __init__(self, train = False):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 48, 7, stride=2),
            nn.ReLU())

        self.conv_2 = nn.Sequential(
            nn.Conv2d(48, 64, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 96, 5),
            nn.ReLU())

        self.conv_4 = nn.Sequential(
            nn.Conv2d(96, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))
            
        
        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 192, 3),
            nn.ReLU())
        
        self.conv_6 = nn.Sequential(
            nn.Conv2d(192, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))

        self.conv_7 = nn.Sequential(
            nn.Conv2d(256, 384, 3),
            nn.ReLU())
        
        self.conv_8 = nn.Sequential(
            nn.Conv2d(384, 512, 3),
            nn.ReLU())
            
        
        self.flatten = nn.Flatten()

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=32769, out_features=4096),
            nn.Dropout(p = 0.5, inplace=train))
        

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p = 0.5, inplace=train))

        self.fc_3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=3074),
            nn.Dropout(p = 0.5, inplace=train))
        
        self.fc_4 = nn.Sequential(
            nn.Linear(in_features=3074, out_features=2048),
            nn.Dropout(p = 0.5, inplace=train))

        self.fc_5 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.Dropout(p = 0.5, inplace=train))
        
        self.fc_6 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=3))



    def forward(self, x: torch.Tensor, speed: torch.Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.flatten(x)
        x = self.fc_1(torch.cat((x, speed), dim=1))
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.fc_4(x)
        x = self.fc_5(x)
        x = self.fc_6(x)
        return x

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net().to(device)
    net.load_state_dict(torch.load('GTA_FSD-1726960744_EPOCH_1.pth', weights_only=True))

    BATCH_SIZE = 5
    summary(net, input_data=(torch.randn(BATCH_SIZE, 3, height, width).to(device), torch.randn(BATCH_SIZE, 1).to(device)))

    data = []
    for i in range(10):
        data.append(torch.rand(BATCH_SIZE, 3, height, width).to(device))

    import time

    start = time.time()
    for i in range(1000):
        net(data[i], torch.rand(BATCH_SIZE,1).to(device))
    end = time.time()
    print(end - start)

