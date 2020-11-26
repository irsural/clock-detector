import sys
import math
import pytesseract
import cv2
import numpy as np
from detector import utilities

from pytesseract import Output


class ClockFace:
    """This class is used for computing working with a clock faceю
    """

    def __init__(self,
                 image=None,
                 minHeighContour=20,
                 maxHeighContour=40,
                 minWidthContour=15,
                 maxWidthContour=40,
                 minCountourArea=250,
                 maxContourArea=700):
        self.image = image

        self.minHeighContour = minHeighContour
        self.maxHeighContour = maxHeighContour

        self.minWidthContour = minWidthContour
        self.maxWidthContour = maxWidthContour

        self.minCountourArea = minCountourArea
        self.maxContourArea = maxContourArea

        self.tesseractConfig = r'--oem 3 --psm 6 outputbase digits'

        self.center = None
        self.radius = None

    def computeClockFace(self):
        filtredImage = self.__filterImageSecond(self.image)

        xCenter = filtredImage.shape[1] // 2
        yCenter = filtredImage.shape[0] // 2

        circles = cv2.HoughCircles(
            filtredImage, cv2.HOUGH_GRADIENT, 1, 1)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = sorted(
                circles, key=lambda circle: circle[2], reverse=True)
            xMax, yMax, rMax = circles[0]

            self.center = (xMax, yMax)
            self.radius = rMax

            return self.__cutImage(self.image, (xMax, yMax), rMax)
        else:
            return self.image

    def computeCentralVector(self, image):
        """Computing the central vector

        Returns:
            turple(centralVector, centralPoint): The computed data.
        """

        thresh = self.__filterImageFirst(image)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        numbers = []

        for cnt in contours:
            areaField = cv2.contourArea(cnt)

            if areaField >= self.minCountourArea and areaField <= self.maxContourArea:
                [x, y, w, h] = cv2.boundingRect(cnt)

                # accuracy = 0.03 * cv2.arcLength(cnt, True)
                # approx = cv2.approxPolyDP(cnt, accuracy, True)
                # cv2.drawContours(image, [approx], 0, (0, 255, 0), 1)
                # cv2.imshow('Approximate Contours', image)
                # if cv2.waitKey() == 27:
                #     sys.exit()

                if w >= self.minWidthContour and w <= self.maxWidthContour \
                        and h >= self.minHeighContour and h <= self.maxHeighContour:
                    isDigit, digit = self.__isDigit(thresh[y:y+h, x:x+w])

                    if not isDigit or digit not in [3, 9]:
                        continue

                    pointCenter = ((2 * x + w) // 2, (2 * y + h) // 2)
                    numbers.append((digit, areaField, pointCenter))

        max3 = None
        max9 = None

        for i in range(len(numbers)):
            if numbers[i][0] == 3:
                if max3 is None or numbers[i][1] > max3[1]:
                    max3 = numbers[i]
            elif numbers[i][0] == 9:
                if max9 is None or numbers[i][1] > max9[1]:
                    max9 = numbers[i]

        (x1, y1) = max3[2]
        (x2, y2) = max9[2]

        if y1 < y2:
            y1, y2 = y2, y1

        centralPoint = ((x1 + x2) // 2, (y1 + y2) // 2)
        centralVector = (x2 - x1, y2 - y1)

        return centralVector, centralPoint

    def __cutImage(self, image, point, radius):
        """Cuts an image by a clock face.

        Args:
            image (numpy.ndarray): The image that is needed to cut.
            point (turple(int, int)): The cetner of the clock face. 
            radius (int): The radius of the clock face.

        Returns:
            numpy.ndarray: The cut image.
        """

        dx0 = abs(point[0] - radius)
        dy0 = abs(point[1] - radius)
        dx1 = dx0 + 2 * radius
        dy1 = dy0 + 2 * radius

        return image[dy0:dy1, dx0:dx1]

    def __filterImageFirst(self, image):
        """Returns a filtred image

        Args:
            image (numpy.ndarray): The input image

        Returns:
            numpy.ndarray: The filtred image
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return thresh

    def __filterImageSecond(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 20, 90, 110)
        thresholdImage = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
        edges = cv2.Canny(thresholdImage, 50, 200, 255)
        return edges

    def __isDigit(self, field):
        """Checks the field is a digit or not.

        Args:
            field (numpy.ndarray): The field.

        Returns:
            boolean: The field is a digit or not.
        """

        d = pytesseract.image_to_string(field, config=self.tesseractConfig)
        return (True, int(d[0])) if d[0].isdigit() else (False, None)
