# GTA5 FSD

This is a project which creates a self-driving car in GTA V with Pytorch that follows the ingame minimap. The model takes in a 640x160 RGB image as well as the current speed of the car and outputs the steering angle and the amount of throttle and brake.

# Python Files
* data_recorder.py records screen shots as BMP image and inputs of steering angle, throttle and brake.
* data_seletor.py analyzes the balance of the training data.
* gta_v_driver_model.py is model in Pytorch.
* train.py trains the model and save model after each epoch.
* gta_fsd_driver.py runs the model, makes prediction on each frame and uses vJoy as input.



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




* The size of model: It is hard to know if the model is big enough or too big. I starts from a smaller model (fewer convolutional and dense layer)