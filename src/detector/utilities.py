import cv2
import numpy as np
import sys
import imutils

def rotate_image(image, angle):
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

def gaussian_array_blur(array):
    """Smoothes the edges of an array.

    Args:
        array ([int, int, ...]): The input array.

    Returns:
        [int, int, ...]: The smoothed array.
    """

    for i in np.arange(1, len(array) - 1, 1):
        array[i] = (array[i - 1] + array[i + 1]) // 2
    return array

def search_local_extremes(graph, min_height=20, min_dist=10):
    """Searches all local extremes with counting the length of extremes.

    Args:
        graph ([int, int, ...]): The input graph (array).
        min_height (int): The minimal extreme's height.
        min_dist (int): The minimal distance between extremes for that they can be considered extremes.

    Returns:
        [(x, y, length), ...]: The array of extremes.
    """

    extremes = []

    max_height = 0
    max_x = 0
    prev_x = 0
    length = 0

    upper = False
    downer = False

    for idx, height in enumerate(graph):
        if height > max_height:
            max_height = height
            max_x = idx
            upper = True
            length += 1
        else:
            if upper:
                if max_height > min_height and (abs(prev_x - max_x) > min_dist or prev_x == 0):
                    extremes.append((max_x, max_height, length))
                    prev_x = max_x
                # TODO: Should I use below code? 
                # elif max_height > min_height and abs(prev_x - max_x) <= min_dist and prev_x != 0:
                #     x, h, l = extremes[len(extremes) - 1]
                #     max_x = (max_x + x) // 2
                #     max_height = (max_height + h) // 2
                #     length = length + l
                #     extremes[len(extremes) - 1] = (max_x, max_height, length)

            max_height = 0
            height = 0
            upper = False
            length = 0

    if upper:
        if max_height > min_height and (abs(prev_x - max_x) > min_dist or prev_x == 0):
            extremes.append((max_x, max_height, length))

    return extremes