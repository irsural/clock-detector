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
                        if 0 <= temp_s <= 5 or \
                           28 <= temp_s <= 32 or \
                           58 <= temp_s <= 60.5:
                           temp_m = reader.get_minutes_by_graph(frame)
                        else:
                            temp_m = reader.get_minutes_by_lines(frame)

                        if temp_m is None:
                            continue

                        if temp_m >= 29.8:
                            temp_m = 0.0

                        if 58 <= temp_s <= 60.5 and temp_m > 0:
                            temp_m -= 1

                        minutes.append(temp_m)
                        seconds.append(temp_s)
                        count += 1

                mean_second = np.median(seconds)
                mean_second = 0.2*round(mean_second/0.2)

                if 59.9 <= (mean_second) <= 60.5:
                    mean_second = 0

                minute = np.median(minutes)

                print('------------------------')
                print(f'minutes: {minutes}')
                print(f'second: {seconds}')
                print(f'time: {int(minute)}:{toFixed(mean_second, 1)}')
            except Exception as e:
                print(e)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break


read_video_lines()
