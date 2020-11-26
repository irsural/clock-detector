import cv2
import numpy as np
import sys
import math
import imutils

from matplotlib import pyplot as plt

from detector import clock_face as cf
from detector import utilities as util


class DetectorTemplate:
    """Detector of a template in a clock face's image.
    """

    def __init__(self, template, method=cv2.TM_SQDIFF_NORMED):
        self.template = template
        self.method = method

    def detect(self, image, radius=None, center=None):
        """Detects a template in an image.
            It detects by using OpenCV's matchTemplate().

        Args:
            image (numpy.ndarray): The input image.
            radius (double, optional): The radius of a clock face. Defaults to None.
            center (double, optional): The point of the clock face's centre. Defaults to None.

        Returns:
            turple(len(cnts), val, loc):
                len(cnts) - Count of found contours after substruction the template and the image.
                val - The value of similirity the template and a found region.
                loc - The start point (x0, y0) of the found region.
        """

        selectedParts = []

        for angle in np.arange(0, 360, 1):
            rotated, mask = util.rotateImage(self.template, angle)
            cv2.imshow('rotated', rotated)

            val, loc, region = DetectorTemplate.findTemplate(
                image, rotated, self.method
            )

            copy_image = image.copy()
            cv2.rectangle(
                copy_image, loc, (loc[0] + region.shape[1], loc[1] + region.shape[0]), (0, 0, 255), 2)
            cv2.imshow('image', copy_image)
            cv2.waitKey(2)

            try:
                im = cv2.bitwise_and(region, region, mask=mask)
                im = cv2.absdiff(im, rotated)
                edged = cv2.Canny(im, 255, 255)

                # I don't know why it needs to define contours, but
                # If sorting but them everything works fine
                cnts, _ = cv2.findContours(
                    edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                selectedParts.append((len(cnts), val, loc, region))

            except Exception as ex:
                print(ex)

        if len(selectedParts) > 0:
            selectedParts = self.__sortSelectedParts(selectedParts)
            return selectedParts[0]

        return None

    def detectWithoutMask(self, image, show_detected=False):
        selectedParts = []

        for angle in np.arange(0, 360, 1):
            rotated = imutils.rotate_bound(self.template, angle)

            val, loc, region = DetectorTemplate.findTemplate(
                image, rotated, self.method
            )

            if show_detected:
                copy_image = image.copy()
                cv2.rectangle(
                    copy_image, loc, (loc[0] + region.shape[1], loc[1] + region.shape[0]), (0, 0, 255), 2)

                cv2.imshow('rotated', rotated)
                cv2.imshow('image', copy_image)
                cv2.waitKey(2)

            selectedParts.append((val, loc, region))

        if len(selectedParts) > 0:
            selectedParts = self.__sortSelectedParts(selectedParts)
            return selectedParts[0]

        return None

    def __sortSelectedParts(self, selected):
        # Sort by val
        selected = sorted(
            selected, key=lambda part: abs(part[0]), reverse=True)
        return selected

    @staticmethod
    def findTemplate(image, template, method):
        """Finds a template in an image.

        Args:
            image (numpy.ndarray): The input image.
            template (numpy.ndarray): The template.
            method (int): The foundition's method.

        Returns:
            turple(
                val,
                loc,
                region
            ):
                val is the similarity coefficient.
                loc is the x0, y0 of a part where the template was found.
                region is the part of the image where the template was found.
        """

        result = cv2.matchTemplate(image, template, method)

        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = minLoc
            val = 1 - minVal
        else:
            loc = maxLoc
            val = maxVal

        region = image[loc[1]:loc[1] + template.shape[0],
                       loc[0]:loc[0] + template.shape[1]]

        return val, loc, region
