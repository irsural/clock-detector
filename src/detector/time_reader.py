import cv2
import config
import numpy as np
import math

from collections import Counter

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
        filtred_image, s_filtred_image = self.prepare_image(
            self.image, is_clock=False, blurring=11)

        graph = TimeReader.convert_image_to_graph(filtred_image)
        graph = utilities.gaussian_array_blur(graph)
        s_graph = TimeReader.convert_image_to_graph(s_filtred_image)
        s_graph = utilities.gaussian_array_blur(s_graph)

        extremes = utilities.search_local_extremes(graph, min_dist=10, min_height=40)
        extremes = sorted(extremes, key=lambda point: point[1], reverse=True)
        s_extremes = utilities.search_local_extremes(s_graph, min_dist=25, min_height=40)
        s_extremes = sorted(s_extremes, key=lambda point: point[2], reverse=False)

        seconds = config.DEGREE / 60
        s_seconds = config.DEGREE / 30

        if len(extremes) >= 1:
            s = extremes[0][0]
            s_s = s_extremes[0][0]

            return int(s_s / s_seconds), s / seconds
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
            extremes = sorted(
                extremes, key=lambda point: point[2], reverse=False)

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

    @staticmethod
    def convert_image_to_graph(image):
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

    def prepare_image(self, image, is_clock, blurring=config.CLOCK_FACE_DEFAULT_BLURRING):
        """Prepares an image by the future reading the time.

        Args:
            image (numpy.ndarray): The input image.
            is_clock (bool, optional): Defines there is a analog clock of a timer in the image.

        Returns:
            numpy.ndarray: The prepared image.
        """

        min_radius = config.MIN_RADIUS_CLOCK if is_clock else config.MIN_RADIUS_TIMER
        max_radius = config.MAX_RADIUS_CLOCK if is_clock else config.MAX_RADIUS_TIMER

        face = cf.ClockFace(image, min_radius=min_radius,
                            max_radius=max_radius, blurring=blurring)
        cut_image, centre, radius = face.search_clock_face()
        rotated = cf.ClockFace.wrap_polar_face(cut_image)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        if is_clock:
            kernel = np.ones((3, 1), np.uint8)
            erosion = cv2.erode(gray, kernel, iterations=4)

            blur = cv2.medianBlur(erosion, 3)
            _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY)

            return thresh
        else:
            # filter the main part
            blur = cv2.medianBlur(gray, 1)
            _, thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV)

            if __debug__:
                cv2.imshow('rotated', rotated)

            # filter the additional (smaller) part
            s_face = cf.ClockFace(cut_image,
                                  min_radius=config.SMALL_TIMER_MIN_RADIUS,
                                  max_radius=config.SMALL_TIMER_MAX_RADIUS,
                                  blurring=1,
                                  by_canny=True)
            s_cut_image, centre, radius = s_face.search_clock_face()

            # if __debug__:
            #     plt.imshow(s_cut_image)
            #     plt.show()

            s_rotated = cf.ClockFace.wrap_polar_face(
                s_cut_image, error_height=30)
            s_gray = cv2.cvtColor(s_rotated, cv2.COLOR_RGB2GRAY)
            s_blur = cv2.medianBlur(s_gray, 13)
            _, s_thresh = cv2.threshold(
                s_blur, 125, 255, cv2.THRESH_BINARY_INV)

            # if __debug__:
            #     plt.imshow(s_rotated)
            #     plt.show()

            #     plt.imshow(s_thresh)
            #     plt.show()

            return thresh, s_thresh
