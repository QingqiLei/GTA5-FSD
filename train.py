

import numpy as np
import cv2
import sys
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from gta_v_driver_model import width
from gta_v_driver_model import height
from gta_v_driver_model import Net
import utils
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on the GPU')
else:
    device = torch.device('cpu')
    print('running on the CPU')
        
import os
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        feature = self.img_labels.iloc[idx, 1].astype(float)
        label = self.img_labels.iloc[idx, 2:].values.astype(float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.Tensor(image), torch.Tensor([feature]), torch.Tensor(label)

class Trainer():
    net = Net(train=True).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    MODEL_NAME = f"GTA_FSD-{int(time.time())}"

    def fwd_pass(self, X_img, X_speed, Y):
        self.net.zero_grad()
        outputs = self.net(X_img, X_speed)
        loss = self.loss_function(outputs, Y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def train(self, dataloader):
        EPOCHS = 5
        with open('model.log', 'a') as f:
            for epoch in range(EPOCHS):
                for batch, (image, feature, label) in enumerate(tqdm(dataloader)):
                    loss = self.fwd_pass(image.to(device), feature.to(device), label.to(device))
                    f.write(f"{self.MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss),5)}\n")
                    f.flush()
                torch.save(self.net.state_dict(),
                        f"{self.MODEL_NAME}_EPOCH_{str(epoch)}_BATCH_{str(batch)}.pth")


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

    
    my_dataset = CustomImageDataset(os.path.join(utils.data_dir, 'tmp.csv'), utils.data_dir, transform=transform)
    print(type(my_dataset.__getitem__(0)[0]))
    # NUM_WORKERS = int(os.cpu_count())
    # dataloader = DataLoader(my_dataset, batch_size=16, shuffle=True, num_workers=NUM_WORKERS)
    # trainer = Trainer()
    # trainer.train(dataloader)
