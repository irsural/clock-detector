import cv2
import config
import numpy as np
import config
import math
import sys

from detector import clock_face as cf
from detector import utilities as util

if __debug__:
    from matplotlib import pyplot as plt

class TimeReader:
    def __init__(self, image):
        self.image = image

    def read(self):
        filtred_image = self.prepare_image(self.image)

        if __debug__:
            plt.imshow(filtred_image, cmap='gray', vmin=0, vmax=255)
            plt.show()

        # I don't fucking know what parameters need here
        lines = cv2.HoughLinesP(filtred_image, rho=3, theta=np.pi / 180,
                                threshold=5, minLineLength=15, maxLineGap=0)
        # soring by x1
        lines = sorted(lines, key=lambda line: line[0][0])
        graph = [(line[0][0], line[0][3]) for line in lines]

        hands = util.find_local_extremes(graph)
        hands = sorted(hands, key=lambda hand: hand[1])  # sorting by length

        if __debug__:
            print(hands)

        if len(hands) >= 3:
            h = hands[2][0]
            m = hands[1][0]
            s = hands[0][0]
        elif len(hands) == 2:
            s = hands[0][0]

            if 50 <= hands[1][1] <= 100:
                h = hands[1][0]
                m = hands[0][0]
            elif 30 <= hands[1][1] < 50:
                h = hands[1][0]
                m = hands[1][0]
            else:
                h = hands[0][0]
                m = hands[1][0] 

        elif len(hands) == 1:
            h = hands[0][0]
            m = hands[0][0]
            s = hands[0][0]

        hours = config.DEGREE / 12
        minutes = config.DEGREE / 60

        return int(h / hours), int(m / minutes), math.ceil(s / minutes)

    def prepare_image(self, image):
        self.face = cf.ClockFace(image)
        cut_image, centre, radius = self.face.compute_clock_face()

        if cut_image is None:
            print('Cannot find a clock face! Aborting...')
            sys.exit()

        if __debug__:
            plt.imshow(cut_image, cmap='gray', vmin=0, vmax=255)
            plt.show()

        rotated = self.face.wrap_polar_face(cut_image)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        if __debug__:
            plt.imshow(rotated, cmap='gray', vmin=0, vmax=255)
            plt.show()

        kernel = np.ones((3, 1), np.uint8)
        erosion = cv2.erode(gray, kernel, iterations=5)
        _, thresh = cv2.threshold(erosion, 125, 255, cv2.THRESH_BINARY)

        return thresh
