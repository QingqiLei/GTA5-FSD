# GTA5 FSD

This is a project which creates a self-driving car in GTA V with Pytorch that follows the ingame minimap. The model takes in a 640x160 RGB image as well as the current speed of the car and outputs the steering angle and the amount of throttle and brake.

# Python Files
* data_recorder.py records screen shots as BMP image and inputs of steering angle, throttle and brake.
* data_seletor.py analyzes the balance of the training data.
* gta_v_driver_model.py is model in Pytorch.
* train.py trains the model and save model after each epoch.
* gta_fsd_driver.py runs the model, makes prediction on each frame and uses vJoy as input.


# Model
The model use 8 convolutional layers and 6 dense layers. The input is the 640x160 RGB image. The output is 3 numbers which are steering angle, throttle and brake.

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [5, 3]                    --
├─Sequential: 1-1                        [5, 48, 77, 317]          --
│    └─Conv2d: 2-1                       [5, 48, 77, 317]          7,104
│    └─ReLU: 2-2                         [5, 48, 77, 317]          --
├─Sequential: 1-2                        [5, 64, 36, 156]          --
│    └─Conv2d: 2-3                       [5, 64, 71, 311]          150,592
│    └─ReLU: 2-4                         [5, 64, 71, 311]          --
│    └─MaxPool2d: 2-5                    [5, 64, 36, 156]          --
├─Sequential: 1-3                        [5, 96, 32, 152]          --
│    └─Conv2d: 2-6                       [5, 96, 32, 152]          153,696
│    └─ReLU: 2-7                         [5, 96, 32, 152]          --
├─Sequential: 1-4                        [5, 128, 15, 75]          --
│    └─Conv2d: 2-8                       [5, 128, 28, 148]         307,328
│    └─ReLU: 2-9                         [5, 128, 28, 148]         --
│    └─MaxPool2d: 2-10                   [5, 128, 15, 75]          --
├─Sequential: 1-5                        [5, 192, 13, 73]          --
│    └─Conv2d: 2-11                      [5, 192, 13, 73]          221,376
│    └─ReLU: 2-12                        [5, 192, 13, 73]          --
├─Sequential: 1-6                        [5, 256, 6, 36]           --
│    └─Conv2d: 2-13                      [5, 256, 11, 71]          442,624
│    └─ReLU: 2-14                        [5, 256, 11, 71]          --
│    └─MaxPool2d: 2-15                   [5, 256, 6, 36]           --
├─Sequential: 1-7                        [5, 384, 4, 34]           --
│    └─Conv2d: 2-16                      [5, 384, 4, 34]           885,120
│    └─ReLU: 2-17                        [5, 384, 4, 34]           --
├─Sequential: 1-8                        [5, 512, 2, 32]           --
│    └─Conv2d: 2-18                      [5, 512, 2, 32]           1,769,984
│    └─ReLU: 2-19                        [5, 512, 2, 32]           --
├─Flatten: 1-9                           [5, 32768]                --
├─Sequential: 1-10                       [5, 4096]                 --
│    └─Linear: 2-20                      [5, 4096]                 134,225,920
│    └─Dropout: 2-21                     [5, 4096]                 --
├─Sequential: 1-11                       [5, 4096]                 --
│    └─Linear: 2-22                      [5, 4096]                 16,781,312
│    └─Dropout: 2-23                     [5, 4096]                 --
├─Sequential: 1-12                       [5, 3074]                 --
│    └─Linear: 2-24                      [5, 3074]                 12,594,178
│    └─Dropout: 2-25                     [5, 3074]                 --
├─Sequential: 1-13                       [5, 2048]                 --
│    └─Linear: 2-26                      [5, 2048]                 6,297,600
│    └─Dropout: 2-27                     [5, 2048]                 --
├─Sequential: 1-14                       [5, 1024]                 --
│    └─Linear: 2-28                      [5, 1024]                 2,098,176
│    └─Dropout: 2-29                     [5, 1024]                 --
├─Sequential: 1-15                       [5, 3]                    --
│    └─Linear: 2-30                      [5, 3]                    3,075
==========================================================================================
Total params: 175,938,085
Trainable params: 175,938,085
Non-trainable params: 0
Total mult-adds (G): 32.41
==========================================================================================
Input size (MB): 6.14
Forward/backward pass size (MB): 162.55
Params size (MB): 703.75
Estimated Total Size (MB): 872.44
==========================================================================================
```
### Versions used

- Python: 3.10
- Pytorch: 2.41

### Pip Packages

These are the pip packages I extracted from pip freeze...

- mss==9.0.2
- numpy==1.26.4
- opencv-python==4.10.0.84
- Pillow==10.4.0
- pygame==2.6.0
- pypiwin32==3.6
- torch==2.4.1
- torchinfo==1.8
- torchvision==0.19.1
- pyvjoystick==1.1.2.1

## Running it


# Thoughts about Self-Driving

* This project makes prediction frame by frame. The problem is there is no speed imformation of other cars on a single frame. We can estimate the speed of other cars through a short video, but not a frame.
For example, My car is following another car at 30 MPH. It will be very easy to tell I am following the car in a video. But it would look like I am going to collide with the front car (don't know it's stationary or moving) in a single frame. And for Stop Sign, the model needs to remember which car comes earlier, so the model should have a short memory. In other workds, it makes prediction on last a few seconds of video.
I am not going to create 3D convolutional model in this project as it's more complecated and training data is hard to collect, the model is only for each frame. So I removed all other vehicles to records training data. In real world self driving, the model should make prediction on the video of past several seconds.
* If a 3D convolutional neural network is used to make prediction on last a few seconds of video, it can aviod recompute in convolutional layer.
* Use high quality driving data, not conflicting data. In my training data, there are some bad quality data.
    1. I drive wrongly sometimes. Some city roads are difficult and I made mistakes to driving on correct roads. I pause the recording and delete the image soon after it happen.
    2. Braking after completing navigation. There were some data that I brake and stop on road after completing the navigation. Those data conflicts with other data that I am pressing accelerator pedal.
Those two kinds of data are conflicting with each other. The model makes prediction on each frame and can not understand why I brakes. So I cleaned my training data. The braking only occurs on turning and trip completion. For example, those two image are about braking on completing the navigation. The navigation disappeared when car is close enough. So, there is no information to tell why I brake here.

<img src="52911.bmp" alt="drawing"/>
<img src="52912.bmp" alt="drawing"/>

