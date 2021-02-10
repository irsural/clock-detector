import cv2
import numpy as np
import imutils
import math

from matplotlib import pyplot as plt


def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate(
        (alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def get_correlation_graph(im1, im2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    result = []
    x = 0
    dx = im1.shape[1]  # h, w, c

    if im1.shape[0] != im2.shape[0]:
        im1 = cv2.resize(
            im1, (im1.shape[1], im2.shape[0]), interpolation=cv2.INTER_AREA)

    while not dx > im2.shape[1]:
        # TODO: St relative values for computing height of image
        # 85 is computed by (100 - 15) <-> (config.DEFAULT_WRAP_POLAR_HEIGHT - config.DEFAULT_WRAP_POLAR_HEIGHT_ERROR)
        part = np.copy(im2[85 - im2.shape[0]:im2.shape[0], x:dx])
        corr = cv2.matchTemplate(im1, part, cv2.TM_CCOEFF_NORMED)[0][0]
        result.append(corr)
        x += 1
        dx += 1

    return result


def find_template(im, template):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    g_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = template.shape

    res = cv2.matchTemplate(g_im, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    part = im[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return part


def blur_graph(graph):
    for i in range(1, len(graph)-1, 1):
        graph[i] = (graph[i-1]+graph[i+1])/2


def rotate_image(im, angle):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 100)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # ensure at least one contour was found
    if len(cnts) > 0:
        # grab the largest contour, then draw a mask for the pill
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # compute its bounding box of pill, then extract the ROI,
        # and apply the mask
        (x, y, w, h) = cv2.boundingRect(c)
        imageROI = im[y:y + h, x:x + w]
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI,
                                   mask=maskROI)

        rotated = imutils.rotate_bound(imageROI, angle)
        return rotated


def coef_corr(im, template):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)[0][0]
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    return res


def distance(point1, point2, point):
    _abs = abs((point2[0]-point1[0])*(point1[1]-point[1])-(point1[0]-point[0])*(point2[1]-point1[1]))
    _sqrt = math.sqrt(math.pow(point2[0]-point1[0], 2) + math.pow(point2[1]-point2[1], 2))
    return _abs / _sqrt;

def distance(point1, point2):
    return math.sqrt(math.pow(point2[0]-point1[0], 2) + math.pow(point2[1]-point1[1], 2))