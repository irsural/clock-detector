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


def filer_response(data):
    data = [round(i, 1) if i % 1 > 0.8 else i for i in data]
    return data


def read_video(is_clock):
    video = cv2.VideoCapture(0)

    while (True):
        key = cv2.waitKey(1) & 0xFF

        ret, frame = video.read()

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        if key == ord('e'):
            minutes = []
            seconds = []

            for i in range(9):
                ret, frame = video.read()
                reader = tr.TimeReader()
                time = reader.get_time_on_clock(
                    frame) if is_clock else reader.get_time_on_timer(frame)

                minutes.append(time[0])
                seconds.append(time[1])

            print('-------------------------')
            print(f'minutes: {minutes}')
            print(f'seconds: {seconds}')
            print(
                f'time: {int(np.median(minutes))}:{toFixed(np.median(seconds), 1)}')
            mean_val = np.mean(seconds)
            round_val = 0.2*round(mean_val/0.2)
            print(
                f'(round_val) time: {int(np.median(minutes))}:{toFixed(round_val, 1)}')
            print(
                f'(mean_val) time: {int(np.median(minutes))}:{toFixed(mean_val, 3)}')
            seconds = filer_response(seconds)
            print(
                f'(my) time: {int(np.median(minutes))}:{toFixed(np.median(seconds), 1)}')
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

# read_video(False)


def read_video_lines():
    video = cv2.VideoCapture(0)

    while (True):
        key = cv2.waitKey(1) & 0xFF

        ret, frame = video.read()
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)

        if key == ord('e'):
            try:
                minutes = []
                seconds = []

                count = 0
                while count < 5:
                    ret, frame = video.read()
                    reader = tr.TimeReader()
                    temp_s = reader.get_seconds_by_lines(frame.copy())
                    if temp_s is not None:
                        if temp_s > 0:
                            temp_m = reader.get_minutes(frame)

                            if (0 <= temp_s <= 16 or temp_s > 58) and temp_m >= 30:
                                temp_m = 0.0

                            if temp_m > 0 and temp_s >= 35:
                                temp_m -= 1

                            if temp_s >= 59.9:
                                temp_s = 0.0

                            minutes.append(temp_m)
                            seconds.append(temp_s)
                            count += 1

                # if 0 <= np.median(seconds) <= 0.4 and np.mean(seconds) > 0.4:
                #     second = np.median(seconds)
                # else:
                #     second = np.mean(seconds)
                second = np.median(seconds)
                second = 0.2*round(second/0.2)

                if 59.9 <= (second) <= 60.5:
                    second = 0

                minute = min(minutes)
                if minute in [30, 15]:
                    minute = 0

                print('------------------------')
                print(f'minutes: {minutes}')
                print(f'second: {seconds}')
                print(f'time: {int(minute)}:{toFixed(second, 1)}')

                # cv2.imshow('lines', reader.get_time_by_lines(frame))
            except Exception as e:
                print(e)
            # cv2.waitKey()

        elif key == ord('q'):
            cv2.destroyAllWindows()
            break


read_video_lines()
