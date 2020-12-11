import cv2
import config
import numpy as np
import config
import math
import sys
import copy

from collections import Counter
from scipy.signal import argrelextrema

from detector import clock_face as cf
from detector import utilities
from detector.exceptions import SearchingHandsError

if __debug__:
    from matplotlib import pyplot as plt


class TimeReader:
    """This class is used for reading time by an image of analog clock.
    """

    def __init__(self, image):
        self.image = image

    def read_timer(self):
        filtred_image = self.prepare_image(self.image, is_clock=False)

        graph = self.convert_image_to_graph(filtred_image)
        graph = utilities.gaussian_array_blur(graph)

        extremes = utilities.search_local_extremes(graph)
        extremes = sorted(extremes, key=lambda point: point[1], reverse=True)

        seconds = config.DEGREE / 60

        if len(extremes) > 1:
            s = extremes[0][0]
            return int(s / seconds)
        else:
            raise SearchingHandsError

    def read_clock(self):
        filtred_image = self.prepare_image(self.image, is_clock=True)

        graph = self.convert_image_to_graph(filtred_image)
        graph = utilities.gaussian_array_blur(graph)

        extremes = utilities.search_local_extremes(graph)

        hours = config.DEGREE / 12
        minutes = config.DEGREE / 60

        if len(extremes) >= 3 or len(extremes) == 2:
            e = sorted(extremes, key=lambda point: point[1], reverse=True)

            if len(extremes) >= 3:
                extremes = [e[0], e[1], e[2]]
            else:
                extremes = e

        if len(extremes) == 3:
            extremes = sorted(extremes, key=lambda point: point[2], reverse=False)

            s = extremes[0][0]
            m = extremes[1][0]
            h = extremes[2][0]
        elif len(extremes) == 2:
            s = extremes[0][0]

            # 1
            if extremes[0][2] > extremes[1][2] * 0.8:
                m = s
                h = extremes[1][0]
            # 2
            elif extremes[0][2] * 1.3 < extremes[1][2]:
                m = extremes[1][0]
                h = m
            # 3
            else:
                h = s
                m = extremes[1][0]
        elif len(extremes) == 0:
            s = extremes[0][0]
            m = s
            h = s
        else:
            raise SearchingHandsError

        # Adding constants to account for the error.
        return int(((h - 5) / hours) + 0.3), int((m - 5) / minutes + 0.3), math.ceil((s - 5) / minutes)

    def convert_image_to_graph(self, image):
        """Builds a graphic by an image.
        The graphic shows count of pixels > 0 in grayscale.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            [int, int, ...]: The output graphic.
        """

        graph = []
        array = []

        height, width = image.shape[:2]

        for x in range(width):
            counter = 0
            for y in range(height):
                if image[y, x] > 0:
                    counter += 1
            graph.append(counter)
            counter = 0

        return graph

    def prepare_image(self, image, is_clock):
        """Prepares an image by the future reading the time.

        Args:
            image (numpy.ndarray): The input image.
            is_clock (bool, optional): Defines there is a analog clock of a timer in the image.

        Returns:
            numpy.ndarray: The prepared image.
        """

        self.face = cf.ClockFace(image, is_clock)
        cut_image, centre, radius = self.face.seach_clock_face()

        if __debug__:
            plt.imshow(cut_image)
            plt.show()

        rotated = self.face.wrap_polar_face(cut_image)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        if is_clock:
            kernel = np.ones((3, 1), np.uint8)
            erosion = cv2.erode(gray, kernel, iterations=4)

            blur = cv2.medianBlur(erosion, 3)
            _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY)
        else:
            blur = cv2.medianBlur(gray, 7)
            _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV)

            if __debug__:
                plt.imshow(thresh, cmap='gray', vmin=0, vmax=255)
                plt.show()

        return thresh