import numpy as np
import cv2
import pandas as pd

from scipy import signal

class Correlator(object):
    @staticmethod
    def correlate(image, template):
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return min_val, max_val, min_loc, max_loc

    @staticmethod
    def phase_correlate(image, template):
        image = np.float32(image)
        template = np.float32(template)

        return cv2.phaseCorrelate(image, template)