# from urllib.request import urlopen
from face_landmark_dnn_vgg_RT import Neural_Network
import time
# import re
# from bs4 import BeautifulSoup
import serial
from cv_close_eye_detect import Eye_Detect
eye_detect = Eye_Detect(0)
nn = Neural_Network()
sp = serial.Serial()
sp.port = 'COM6'
sp.baudrate = 9600
sp.timeout = 1
sp.open()

while True:
    value = sp.readline()  # wait to get alcohol data
    if value == b'alcohol\r\n':  # alcohol test pass
        print("alcohol pass")
        break
    else:
        continue

while True:
    ret, img = eye_detect.get_frames()
    face_flag = nn.test(ret=ret, img=img)
    if face_flag == 252:    # face ID pass
        print("faceID pass")
        sp.write(b'1')     # launch the car
        break
    else:
        continue

cont_close = 0
while True:
    eye_flag = eye_detect.detect()
    if eye_flag == 255:
        # sp.readline()  # detect no eyes, then beep the buzzer & turn on the light
        if cont_close < 3:
            cont_close += 1
        else:
            cont_close = 0
            sp.write(b'2')
    else:
        # sp.readline()  # open eyes, then continue the detection
        # sp.write(b'3')
        cont_close = 0
        continue
