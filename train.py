

import numpy as np
import cv2
import sys
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

from gta_v_driver_model import width
from gta_v_driver_model import height
from gta_v_driver_model import Net
import utils


data_sir = utils.data_sir
speed_fil_path = utils.speed_file


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on the GPU')
else:
    device = torch.device('cpu')
    print('running on the CPU')


class GTADataset():

    data_dir = data_sir

    training_data = []

    imgs = []

    X_img = []
    X_speed = {}
    Y = []
    total_img = 0
    max_speed = 0

    

    def make_training_data(self):
        lines = open(os.path.join(data_sir, 'data.txt'), 'r').readlines()
        for line in lines:
            steering_angle, throttle, brake, speed, path = line.split(',')
            steering_angle, throttle, brake, speed, path = float(steering_angle), float(
                throttle), float(brake), float(speed), path.strip()

            if os.path.isfile(path):
                self.max_speed = max(self.max_speed, speed)
                img = cv2.imread(path)
                self.imgs.append(
                    [path, steering_angle, throttle, brake, speed, img])

        np.random.shuffle(self.imgs)
        self.X_img = torch.Tensor(
            [[i[-1][:, :, 0], i[-1][:, :, 1], i[-1][:, :, 2]] for i in self.imgs])
        print(self.X_img.shape)
        self.X_speed = torch.Tensor(
            [i[-2] / self.max_speed for i in self.imgs]).view(-1, 1)
        self.Y = torch.Tensor([[i[1], i[2], i[3]] for i in self.imgs])
        self.total_img = len(self.imgs)
        print('total data:', len(self.imgs))
        torch.save(self.X_img, 'x_img.pth')
        torch.save(self.X_speed, 'x_speed.pth')
        torch.save(self.Y, 'Y.pth')

    def load(self):
        self.X_img = torch.load('x_img.pth', weights_only=True)
        self.X_speed = torch.load('x_speed.pth', weights_only=True)
        self.Y = torch.load('Y.pth', weights_only=True)
        self.total_img = len(self.imgs)

        


class Trainer():

    net = Net().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    MODEL_NAME = f"GTA_FSD-{int(time.time())}"

    def fwd_pass(self, X_img, X_speed, Y, train=False):
        if train:
            self.net.zero_grad()
        outputs = self.net(X_img, X_speed)
        loss = self.loss_function(outputs, Y)

        if train:
            loss.backward()
            self.optimizer.step()
        return loss

    def train(self, x_img, x_speed, y):
        BATCH_SIZE = 30
        EPOCHS = 30
        print(len(x_img))
        with open('model.log', 'a') as f:
            for epoch in range(EPOCHS):
                for i in tqdm(range(0, len(x_img), BATCH_SIZE)):
                    batch_X_img = x_img[i: i+BATCH_SIZE].to(device)
                    batch_X_speed = x_speed[i: i+BATCH_SIZE].to(device)
                    batch_y = y[i:i+BATCH_SIZE].to(device)
                    loss = self.fwd_pass(
                        batch_X_img, batch_X_speed, batch_y, train=True)

                    if i % BATCH_SIZE == 0:

                        f.write(
                            f"{self.MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss),2)}\n")
                print(f'epoch: {epoch}, loss: {loss}')
                torch.save(self.net.state_dict(),
                           f"{self.MODEL_NAME}_EPOCH_{str(epoch)}.pth")




if __name__ == "__main__":
    gta_dataset = GTADataset()
    # gta_dataset.make_training_data()
    gta_dataset.load()
    trainer = Trainer()
    trainer.train(gta_dataset.X_img.to(device), gta_dataset.X_speed.to(device), gta_dataset.Y.to(device))
