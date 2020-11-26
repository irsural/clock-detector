import cv2
import numpy as np
import sys
import imutils

from math import pow, sqrt, acos, pi


def rotateImage(image, angle):
    """Rotates the main content in an image

    Args:
        image (numpy.ndarray): The image for rotation
        angle (int): The rotation's angle

    Returns:
        numpy.ndarray: The image with the rotated main content
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 170, 220, 0)

    # edged = cv2.Canny(threshold, 20, 100)

    cnts = cv2.findContours(threshold, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_TC89_KCOS)
    cnts = imutils.grab_contours(cnts)

    maskROI = None

    # ensure at least one contour was found
    if len(cnts) > 0:
        # grab the largest contour, then draw a mask for the pill
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)

        # compute its bounding box of pill, then extract the ROI,
        # and apply the mask
        (x, y, w, h) = cv2.boundingRect(c)
        imageROI = image[y:y + h, x:x + w]
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI,
                                   mask=maskROI)

    return imutils.rotate_bound(imageROI, angle), imutils.rotate_bound(maskROI, angle)



def imageFilteringByMasks(image):
    """Filters an image for the future work.

    Args:
        image (numpy.ndarray): The image that will be filtred.

    Returns:
        numpy.ndarray: The filtred image.
    """

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    whiteMask = cv2.inRange(image, lower, upper)

    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellowMask = cv2.inRange(image, lower, upper)

    # combine the mask
    mask = cv2.bitwise_or(whiteMask, yellowMask)

    height, width = mask.shape
    skel = np.zeros([height, width], dtype=np.uint8)  # [height,width,3]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while(np.count_nonzero(mask) != 0):
        eroded = cv2.erode(mask, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        mask = eroded.copy()

    edges = cv2.Canny(skel, 100, 150)

    return edges


def getDistanceBetweenPoints(point1, point2):
    """Returns distance between two points.

    Args:
        point1 (turple(x, y)): The first point.
        point2 (turple(x, y)): The second point.

    Returns:
        double: The distance between two points.
    """

    return sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))


def getDistanceBetweenLineAndPoint(point, pointLine1, pointLine2):
    """Returns distance between a line and a point.

    Args:
        point (turple(x, y)): The point.
        pointLine1 (turple(x, y)): The first point of the line.
        pointLine2 (turple(x, y)): The second point of the line.

    Returns:
        double: The distance between a line and a point.
    """

    line = abs(
        (pointLine2[1] - pointLine1[1]) * point[0] -
        (pointLine2[0] - pointLine1[0]) * point[1] +
        pointLine2[0] * pointLine1[1] - pointLine2[1] * pointLine1[0]
    )
    return line / getDistanceBetweenPoints(pointLine1, pointLine2)


def getAngleBetweenVectors(vector1, vector2):
    """Returns an angle between two vectors.

    Args:
        vector1 (turple(x, y)): The first vector.
        vector2 (turple(x, y)): The second vector.

    Returns:
        double: The angle between two vectors.
    """

    cosAngle = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) /\
        (sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1]) *
         sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1]))
    return (acos(cosAngle) * 180) / pi
