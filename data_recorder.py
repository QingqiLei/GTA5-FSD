import time
import csv
import win32gui
import win32api
import os
import sys
import PIL
from PIL import Image
import mss
import pygame
import utils


save_path = utils.data_dir
label_file = utils.label_file
speed_fil_path = 'speed.txt'

max_samples = 100000
samples_per_second = 7

if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_file = open(label_file, 'a')

print('Press - to start recording!!!!!')

pics = os.listdir(save_path)

total_count = 0

for pic in pics:
	if 'bmp' in pic:
		id = pic.split('.')[0]
		total_count = max(int(id), total_count)
print('current count', total_count)


current_sample = 1
last_time = 0
start_time = time.time()
wait_time = (1/samples_per_second)
stats_frame = 0

sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

pause = True
return_was_down = False
speed = '0.0'

pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x)
             for x in range(pygame.joystick.get_count())]
joysticks[0].init()

previous_image_capture_time = 0

while True:
    pygame.event.pump()

    if (win32api.GetAsyncKeyState(0xBD) & 0x8001 > 0):
        if (return_was_down == False):
            if (pause == False):
                pause = True
            else:
                pause = False

        return_was_down = True
    else:
        return_was_down = False

    if (time.time() - last_time >= wait_time):

        fps = 1 / (time.time() - last_time)
        last_time = time.time()

        if (pause):
            time.sleep(0.01)
            continue

        stats_frame += 1
        if (stats_frame >= 10):
            stats_frame = 0
            os.system('cls')
            print('FPS: %.2f Total Samples: %d Time: %s' % (fps, current_sample,
                  time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
            if (pause == False):
                print('Status: Recording')
            else:
                print('Status: Paused')

        file = open(speed_fil_path, "r")
        new_speed = file.read()
        file.close()

        if (len(new_speed) > 0):
            speed = new_speed

        sct_img = sct.grab(mon)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        img = img.resize((640, 360), PIL.Image.BICUBIC)
        img = img.crop(box=(0, 150, 640, 360))

        steering_angle = joysticks[0].get_axis(0)
        if (abs(steering_angle) < 0.008):
            steering_angle = 0
        steering_angle = (steering_angle+1)/2
        if (steering_angle > 0.99):
            steering_angle = 1
        if (steering_angle < 0.01):
            steering_angle = 0

        throttle = joysticks[0].get_axis(5)
        if (throttle > 0.98):
            throttle = 1
        throttle = 1-(throttle+1)/2

        brake = joysticks[0].get_axis(4)
        if (brake > 0.98):
            brake = 1
        brake = 1-(brake+1)/2

        # print(joysticks[0], round(steering_angle, 2), round(brake, 2), round(throttle,2), speed)
        total_count += 1
        current_sample += 1
        img_name = '%d.bmp' % (total_count)
        img.save(os.path.join(save_path, img_name), 'BMP')
        csv_file.write('%s,%s,%f,%f,%f\n' %
                       (img_name, speed,  steering_angle, throttle, brake))
        csv_file.flush()

        if (current_sample >= max_samples):
            break


print('\nDONE')
print('Total Samples: %d\n' % current_sample)

joysticks[0].quit()
pygame.display.quit()
pygame.joystick.quit()
