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

if __debug__:
    from matplotlib import pyplot as plt


class TimeReader:
    """This class is used for reading time by an image of analog clock.
    """

    def __init__(self, image, is_clock=True):
        self.image = image
        self.is_clock = is_clock

    def read_by_graph(self):
        filtred_image = self.prepare_image(self.image)
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

        if self.is_clock:
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
            else:
                s = extremes[0][0]
                m = s
                h = s

            # Adding constants to account for the error.
            return int(((h - 5) / hours) + 0.3), int((m - 5) / minutes + 0.3), math.ceil((s - 5) / minutes)
        else:
            # TODO: Creating read the time for a timer.
            s = hands[0][0]
            return int(s / minutes)

    def convert_image_to_graph(self, image):
        """Builds a graphic by an image.
        The graphic shows count of pixels > 0 in grayscale.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            [[int, int, ...]]: The output graphic.
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

    def prepare_image(self, image):
        """Prepares an image by the future reading the time.

        Args:
            image (numpy.ndarray): The original image.

        Returns:
            numpy.ndarray: The prepared image.
        """

        self.face = cf.ClockFace(image, self.is_clock)
        cut_image, centre, radius = self.face.seach_clock_face()

        rotated = self.face.wrap_polar_face(cut_image)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        kernel = np.ones((3, 1), np.uint8)
        erosion = cv2.erode(gray, kernel, iterations=4)
        blur = cv2.medianBlur(erosion, 3)
        _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY)

        return thresh
