import cv2
import config
import numpy as np
import config

from detector import clock_face as cf
from detector import utilities as util

class TimeReader:
    def __init__(self, image):
        self.image = image

    def read(self):
        preparedImage = self.prepareImage(self.image)

        lines = cv2.HoughLinesP(preparedImage, rho=3, theta=np.pi / 180,
                                threshold=0, minLineLength=10, maxLineGap=0)
        # soring by x1
        lines = sorted(lines, key=lambda line: line[0][0])
        graph = [(line[0][0], line[0][3]) for line in lines]

        hands = util.getExtremes(graph)
        hands = sorted(hands, key=lambda hand: hand[1])  # sorting by length

        h = hands[2][0]
        m = hands[1][0]
        s = hands[0][0]

        hours = config.DEGREE / 12
        minutes = config.DEGREE / 60

        return int(h / hours), int(m / minutes), int(s / minutes)

    def prepareImage(self, image):
        self.face = cf.ClockFace(image)
        cutImage, centre, radius = self.face.computeClockFace()
        rotated = self.face.wrapPolarImage(cutImage)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        kernel = np.ones((4, 1), np.uint8)
        erosion = cv2.erode(gray, kernel, iterations=6)
        _, thresh = cv2.threshold(erosion, 125, 255, cv2.THRESH_BINARY)

        return thresh
