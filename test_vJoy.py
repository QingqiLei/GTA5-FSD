import utils
import os
import time
from pyvjoystick import vjoy

# Pythonic API, item-at-a-time
j = vjoy.VJoyDevice(1)


while True:

    vjoy_max = utils.vjoy_max

    time.sleep(3)

    j._data.wAxisX = int(vjoy_max * 0.5)

    j._data.wAxisY = int(vjoy_max * 1)  # 0 is biggest

    j._data.wAxisZ = int(vjoy_max * 1)
    j.update()
    os.system('cls')
    print('steer: %.2f thr: %.2f brake: %.2f' % (j._data.wAxisX /
          vjoy_max, j._data.wAxisY/vjoy_max, j._data.wAxisZ/vjoy_max))

    time.sleep(3)

    j._data.wAxisX = int(vjoy_max * 1)  # right

    # j._data.wAxisY = int(vjoy_max * 0)

    # j._data.wAxisZ = int(vjoy_max * 0)

    j.update()
    os.system('cls')
    print('steer: %.2f thr: %.2f brake: %.2f' % (j._data.wAxisX /
          vjoy_max, j._data.wAxisY/vjoy_max, j._data.wAxisZ/vjoy_max))
