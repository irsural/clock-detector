import cv2
import numpy as np
import sys
import math
import imutils

from matplotlib import pyplot as plt

from clockFace import ClockFace
from utilities import getDistanceBetweenPoints, rotateImage, imageFilteringByMasks


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
            rotated, mask = rotateImage(self.template, angle)
            cv2.imshow('rotated', rotated)

            val, loc, region = DetectorTemplate.findTemplate(
                image, rotated, self.method
            )

            copy_image = image.copy()
            cv2.rectangle(copy_image, loc, (loc[0] + region.shape[1], loc[1] + region.shape[0]), (0, 0, 255), 2)
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

    def detectBySIFT(self, image, show_detected=False):
        """Detects a template in an image by SIFT.

        Args:
            image (numpy.ndarray): The input image.
            show_detected (bool, optional): Defines if it needs to show a found region. Defaults to False.
        """

        MIN_MATCH_COUNT = 10

        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_image, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=100)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            area = cv2.contourArea(np.int32(dst))
            area_tempalate = self.template.shape[0] * self.template.shape[1]

            if area < area_template or area > 2 * area_template:
                return

            if len(np.uint32(dst)) != 4:
                return

            image = cv2.polylines(
                image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - %d/%d" %
                  (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        if show_detected:
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                            singlePointColor=None,
                            matchesMask=matchesMask,  # draw only inliers
                            flags=2)

            img3 = cv2.drawMatches(self.template.copy(), kp1, image, kp2,
                                good, None, **draw_params)
            plt.imshow(img3, 'gray'), plt.show()

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
                val is the similarity coefficientю
                loc is the x0, y0 of a part where the template was found/
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
