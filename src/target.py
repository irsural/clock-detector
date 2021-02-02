import cv2
import numpy as np
import pandas as pd
import config

from matplotlib import pyplot as plt
from detector import clock_face as cf
from detector import time_reader as tr
from detector import utilities

def read_video(is_clock):
    video = cv2.VideoCapture(0)
    timeReader = tr.TimeReader()

    while(True):
        # try:
        ret, frame = video.read()

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_path = '../resources/timer/hand.png'
            template = utilities.read_transparent_png(template_path)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            w, h = template.shape[::-1]

            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

            threshold = 0.8
            loc = np.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

            cv2.imshow('Detected', frame)
            cv2.waitKey()
#                 minutes = []
#                 seconds = []

#                 for i in range(5):
#                     time = timeReader.get_time_on_clock(frame) if is_clock else timeReader.get_time_on_timer(frame)
#                     minutes.append(time[0])
#                     seconds.append(time[1])
#                 print(f'{np.median(minutes)}:{np.median(seconds)}')

                # time = timeReader.get_time_on_clock(frame) if is_clock else timeReader.get_time_on_timer(frame)
                # for idx, i in enumerate(time):
                #     ii = str(i)
                #     if len(ii) == 1:
                #         ii = f'0{ii}'
                #     if idx == len(time)-1:
                #         print(ii)
                #     else:
                #         print(ii, end=':')
        # except Exception as e:
        #     print('Не могу распознать циферблат!')
        #     print(f'Исключение: {e}')

read_video(False)

# def func():
#     img_rgb = read_video()
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#     template_path = '../resources/timer/hand.png'
#     template = utilities.read_transparent_png(template_path)
#     w, h = template.shape[::-1]

#     res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

#     threshold = 0.8
#     loc = np.where(res >= threshold)

#     for pt in zip(*loc[::-1]):
#         cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

#     cv2.imshow('Detected', img_rgb)
#     cv2.waitKey()

# func()