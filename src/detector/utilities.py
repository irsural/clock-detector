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

def find_local_extremes(graph, dist=5):
    """Detect all locals extremes on a graph.

    Args:
        graph ([(x, y)]): Points on the graph.
        dist (int): The max possible distance between points by x.

    Returns:
        [(x, y)]: All locals extremes on the graph.
    """

    extremes = []
    x, y = graph[0]

    for i in np.arange(1, len(graph), 1):
        if graph[i][0] - graph[i - 1][0] <= dist:
            if y > graph[i][1]:
                x, y = graph[i]
        else:
            extremes.append((x, y))
            x, y = graph[i]

    extremes.append((x, y))
    return extremes