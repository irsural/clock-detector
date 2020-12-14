import copy
import cv2
import numpy as np
import config

from matplotlib import pyplot as plt

from detector import utilities
from detector.exceptions import SearchingClockFaceError

if __debug__:
    from matplotlib import pyplot as plt


class ClockFace:
    """This class is used for computing working with a clock face.
    """

    def __init__(self, image, min_radius, max_radius,
                 blurring=config.CLOCK_FACE_DEFAULT_BLURRING,
                 by_canny=False):
        self.image = image
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.center = None
        self.radius = None
        self.blurring = blurring
        self.by_canny = by_canny

    def search_clock_face(self):
        """Searches a clock face in an image.

        Raises:
            ClockFaceSearchingError: Raise if a clock face was not found.

        Returns:
            turple(cut_image, (x, y), r):
                cut_image is an image that cut by the circle's boards.
                (x, y) - the central point of the clock face.
                r - the radius of the clock face.
        """

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, self.blurring)

        if not self.by_canny:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                1,
                config.CLOCK_MIN_DIST,
                param1=config.CLOCK_PARAM_1,
                param2=config.CLOCK_PARAM_2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
        else:
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(blurred, kernel, iterations=1)
            canny = cv2.Canny(erosion, 100, 200)

            circles = cv2.HoughCircles(
                canny,
                cv2.HOUGH_GRADIENT,
                2,
                config.TIMER_MIN_DIST,
                param1=config.CLOCK_PARAM_1,
                param2=config.CLOCK_PARAM_2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]

            self.center = (x, y)
            self.radius = r

            cutImage = ClockFace.cut_image(self.image.copy(), (x, y), r)

            return cutImage, (x, y), r
        else:
            raise SearchingClockFaceError

    @staticmethod
    def wrap_polar_face(image,
                        width=config.DEFAULT_WRAP_POLAR_WIDTH,
                        height=config.DEFAULT_WRAP_POLAR_HEIGHT,
                        error_height=config.DEFAULT_WRAP_POLAR_HEIGHT_ERROR):
        """Wraps the clock face up.

        Args:
            image (numpy.ndarray): The clock face image.
            width (int, optional): The width of the final image. Defaults to config.WRAP_POLAR_WIDTH.
            height (int, optional): The height of the final image. Defaults to config.WRAP_POLAR_HEIGHT.
            error_height (int, optional): The error of the height. Defaults to config.DEFAULT_WRAP_POLAR_HEIGHT_ERROR

        Returns:
            numpy.narray: The final wrapped image.
        """

        rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        centre = rotate.shape[0] // 2

        polarImage = cv2.warpPolar(rotate, (height, width), (centre, centre), centre,
                                   cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)

        cropImage = copy.deepcopy(polarImage[0:width, error_height:height])
        cropImage = cv2.rotate(cropImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return cropImage

    @staticmethod
    def cut_image(image, point, radius):
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
