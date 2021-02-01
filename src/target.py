import cv2
import numpy as np
import pandas as pdddd
import config

from matplotlib import pyplot as plt
from detector import clock_face as cf
from detector import time_reader as tr

def read_video(is_clock):
    video = cv2.VideoCapture(0)
    timeReader = tr.TimeReader()
    
    while(True):
        try:
            ret, frame = video.read()
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1)

            if key == ord('e'):
                minutes = []
                seconds = []
                
                for i in range(5):
                    time = timeReader.get_time_on_clock(frame) if is_clock else timeReader.get_time_on_timer(frame)
                    minutes.append(time[0])
                    seconds.append(time[1])
#                 for idx, i in enumerate(time):
#                     ii = str(i)
#                     if len(ii) == 1:
#                         ii = f'0{ii}'
#                     if idx == len(time)-1:
#                         print(ii)
#                     else:
#                         print(ii, end=':')

                print(f'{np.median(minutes)}:{np.median(seconds)}')

            if key == ord('q'):
                return
        except Exception as e:
            print(e)
        
read_video(False)