import copy
import math
import sys
import cv2
import numpy as np
import pytesseract

from matplotlib import pyplot as plt
from pytesseract import Output
from detector import utilities

import config


class ClockFace:
    """This class is used for computing working with a clock faceю
    """

    def __init__(self, image=None):
        self.image = image
        self.center = None
        self.radius = None

    def compute_clock_face(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 25)

        if __debug__:
            plt.imshow(blurred, cmap='gray', vmin=0, vmax=255)
            plt.show()

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1,
            config.CLOCK_FACE_MIN_DIST,
            param1=config.CLOCK_FACE_PARAM_1,
            param2=config.CLOCK_FACE_PARAM_2,
            minRadius=config.CLOCK_FACE_MIN_RADIUS,
            maxRadius=config.CLOCK_FACE_MAX_RADIUS
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]

            self.center = (x, y)
            self.radius = r

            cutImage = self.cut_image(self.image.copy(), (x, y), r)

            return cutImage, (x, y), r
        else:
            return None, None, None

    def wrap_polar_face(self, image, width=config.WRAP_POLAR_WIDTH,
                       height=config.WRAP_POLAR_HEIGHT):
        rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        centre = rotate.shape[0] // 2

        polarImage = cv2.warpPolar(rotate, (height, width), (centre, centre), centre,
                                   cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)

        cropImage = copy.deepcopy(polarImage[0:width, 15:height])
        cropImage = cv2.rotate(cropImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return cropImage

    def cut_image(self, image, point, radius):
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
