import cv2
import numpy as np
import pandas as pd
import config

from matplotlib import pyplot as plt
from detector import clock_face as cf
from detector import time_reader as tr
from detector import utilities

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

def read_video(is_clock):
    video = cv2.VideoCapture(0)

    while (True):
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            # ret, frame = video.read()
            # reader = tr.TimeReader()
            # im = reader.find_lines(frame)
            # cv2.imshow('lines', im)
            minutes = []
            seconds = []

            for i in range(10):
                ret, frame = video.read()
                reader = tr.TimeReader()
                time = reader.get_time_on_clock(frame) if is_clock else reader.get_time_on_timer(frame)

                minutes.append(time[0])
                seconds.append(time[1])

            print('-------------------------')
            print(f'minutes: {minutes}')
            print(f'seconds: {seconds}')
            print(f'time: {np.median(minutes)}:{np.median(seconds)}')
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            ret, frame = video.read()
            cv2.imshow('frame', frame)

read_video(False)