
import win32api
from gta_v_driver_model import Net
from PIL import Image
from pyvjoystick import vjoy
import PIL
import time
import mss
import os
import numpy as np
import torch
from torchvision.transforms import ToTensor
import utils
import numpy
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
else:
    device = torch.device('cpu')
    print('running on CPU')


net = Net().to(device)
net.load_state_dict(torch.load('GTA_FSD-1726960744_EPOCH_2.pth', weights_only=True))
print(net.type)

joy = vjoy.VJoyDevice(1)

class FPSTimer:
    def __init__(self):
        self.t = time.time()
        self.iter = 0

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            print('FPS: %0.2f' % (100.0 / (e - self.t)))
            self.t = time.time()
            self.iter = 0


def predict_loop():
    pause = True
    return_was_down = False
    sct = mss.mss()
    mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    speed = 0
    print('Ready')

    while True:

        if (win32api.GetAsyncKeyState(0xBD) & 0x8001 > 0):
            if (return_was_down == False):
                if (pause == False):
                    pause = True
                    joy._data.wAxisX = int(vjoy_max * 0.1)
                    joy._data.wAxisY = int(vjoy_max * 0)
                    joy._data.wAxisZ = int(vjoy_max * 1)
                    joy.update()

                    print('Paused')
                else:
                    pause = False

                    print('Resumed')

            return_was_down = True
        else:
            return_was_down = False

        if (pause):
            time.sleep(0.01)
            continue

        sct_img = sct.grab(mon)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img = img.resize((640, 360), PIL.Image.BICUBIC)
        img = img.crop(box=(0, 200, 640, 360))

        img = Image.open('data2/4.bmp')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])


        x = transform(img)

        try:
            file = open("speed.txt", "r")
            speed = float(file.read())
            file.close()
        except (ValueError):
            pass

        img = torch.Tensor(numpy.array([x.numpy()])).to(device)
        speed = torch.Tensor([[speed]]).to(device)

        predictions = net(img, speed)
        exit()
        predictions = predictions[0]

        vjoy_max = utils.vjoy_max

        steering_anble = min(max(predictions[0], 0), 1)
        Throttle = min(max(predictions[1], 0), 1)
        Brake = min(max(predictions[2] , 0), 1)

        joy._data.wAxisX = int(vjoy_max * steering_anble)
        joy._data.wAxisY = int(vjoy_max *  Throttle)
        joy._data.wAxisZ = int(vjoy_max *  Brake)
        joy.update()

        os.system('cls')
        print("Steering Angle: %.2f" %steering_anble)
        print("Throttle: %.2f" % Throttle)
        print("Brake: %.2f" % Brake)
        print("speed: %s" % speed)


if __name__ == "__main__":
    predict_loop()
