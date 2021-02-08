import cv2
import config
import numpy as np

from detector import clock_face as cf
from detector import utilities
from matplotlib import pyplot as plt

from icecream import ic


class TimeReader:
    """This class is used for reading time by an image of analog clock.
    """

    def __init__(self, path_templates='../resources/'):
        self.clock_templates = [
            utilities.read_transparent_png(path_templates+'clock/face.png'),
            utilities.read_transparent_png(path_templates+'clock/second.png'),
        ]

        self.timer_templates = [
            utilities.read_transparent_png(path_templates+'timer/face.png'),
            utilities.read_transparent_png(path_templates+'timer/hand.png'),
            utilities.read_transparent_png(
                path_templates+'timer/small_face.png'),
            utilities.read_transparent_png(
                path_templates+'timer/small_hand.png'),
            utilities.read_transparent_png(
                path_templates+'timer/hand_black.png'),
        ]

    def get_time_on_timer(self, im):
        rotated = TimeReader.get_rotated(im, self.timer_templates[0])

        s_rotated = utilities.find_template(
            rotated[0], self.timer_templates[2])
        s_rotated = cf.ClockFace.wrap_polar_face(
            s_rotated, width=360, error_height=30)
        s_part = np.copy(s_rotated[0:s_rotated.shape[0], 0:15])
        s_rotated = np.concatenate((s_rotated, s_part), axis=1)

        rotated = rotated[1]
        part = np.copy(rotated[0:rotated.shape[0], 0:15])
        rotated = np.concatenate((rotated, part), axis=1)

        g_hand = utilities.get_correlation_graph(
            self.timer_templates[1], rotated)
        g_s_hand = utilities.get_correlation_graph(
            self.timer_templates[4], s_rotated)

        # plt.plot(g_s_hand), plt.show()

        pos_hand = g_hand.index(max(g_hand))
        if config.DEFAULT_WRAP_POLAR_WIDTH - config.DEFAULT_WRAP_POLAR_WIDTH*0.83 <= pos_hand <= config.DEFAULT_WRAP_POLAR_WIDTH:
            max_pos_hand = 0
            for i in range(int(config.DEFAULT_WRAP_POLAR_WIDTH // 2.4), int(config.DEFAULT_WRAP_POLAR_WIDTH // 1.8), 1):
                if g_hand[i] >= 0.35 and g_hand[i] >= max_pos_hand:
                    pos_hand = i
                    max_pos_hand = i

        pos_s_hand = g_s_hand.index(max(g_s_hand))
        pos_hand += self.timer_templates[1].shape[1] // 2 + 3
        pos_s_hand += self.timer_templates[3].shape[1] // 2 + 3

        pos_hand %= config.DEFAULT_WRAP_POLAR_WIDTH
        pos_s_hand %= config.DEFAULT_WRAP_POLAR_WIDTH

        if ((pos_s_hand * 30 // 360) in [14, 15] and (pos_hand * 60 // config.DEFAULT_WRAP_POLAR_WIDTH) == 60) or \
            (0 <= pos_hand * 60 // config.DEFAULT_WRAP_POLAR_WIDTH <= 5 and (pos_s_hand * 30 // 360) == 30):
            pos_s_hand = 0

        s = pos_hand * 60 / config.DEFAULT_WRAP_POLAR_WIDTH
        m = pos_s_hand * 30 // 360

        return (int(m), s)

    def get_time_on_clock(self, im):
        _, rotated = TimeReader.get_rotated(im, self.clock_templates[0])
        part = np.copy(rotated[0:rotated.shape[0], 0:15])
        rotated = np.concatenate((rotated, part), axis=1)

        # Find second
        ######################################
        g_second = utilities.get_correlation_graph(
            self.clock_templates[1], rotated)
        utilities.blur_graph(g_second)
        s = g_second.index(min(g_second)) + \
            self.clock_templates[1].shape[1] / 2
        s %= 360

        # Find hour & minute
        ######################################
        rotated = rotated[0:rotated.shape[0]-10, 0:rotated.shape[1]]
        filtred_im = self.filter_image_(rotated)

        graph = TimeReader.convert_image_to_graph(filtred_im)

        # Refund for inverting the reverse side of the second hand
        dis = s + 176
        summer = 1
        for i in range(len(graph)):
            i %= 360

            if (dis - 15) % 360 <= i <= (dis + 15) % 360:
                if graph[i] > 2:
                    graph[i] += summer

                if i <= (dis % 360):
                    summer += 1
                else:
                    summer -= 1

        utilities.blur_graph(graph)

        kp = [(x, y) for x, y in enumerate(graph) if y > 12]

        groups = self.sum_groups_value_(self.group_points_(kp))
        filtred_groups = self.remove_useless_groups_(groups)

        m = max(filtred_groups, key=lambda p: p[1])[0]
        if len(filtred_groups) >= 2:
            h = min(filtred_groups, key=lambda p: p[1])[0]
        else:
            h = m

        h = (h * 12) // config.DEFAULT_WRAP_POLAR_WIDTH
        m = (m * 60) // config.DEFAULT_WRAP_POLAR_WIDTH
        s = (s * 60) // config.DEFAULT_WRAP_POLAR_WIDTH

        return (int(h), round(m), round(s))

    def remove_useless_groups_(self, graph):
        n = []
        for x, y, l in graph:
            if y < 40:
                n.append((x, y, l))
            else:
                add = True
                for i in range(len(n)):
                    _x, _y, _l = n[i]

                    if _y >= 40:
                        if y > _y:
                            n[i] = (x, y, l)
                        add = False
                        break

                if add:
                    n.append((x, y, l))
        return n

    def sum_groups_value_(self, groups):
        result = []

        for group in groups:
            max_x = 0
            max_y = 0

            for x, y in group[0]:
                if y > max_y:
                    max_y = y
                    max_x = x

            result.append((max_x, max_y, len(group[0])))

        return result

    def group_points_(self, points):
        group_points = []
        group = [points[0]]
        prev_x = points[0][0]
        length = 1

        for i in range(1, len(points), 1):
            if (points[i][0] - 3 > prev_x):
                group_points.append((group.copy(), length))
                length = 1
                group.clear()
                group = [points[i]]
            else:
                length += 1
                group.append(points[i])

            prev_x = points[i][0]

        group_points.append((group.copy(), length))

        return group_points

    def filter_image_(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kernel = (5, 5)
        erosion = cv2.erode(gray, kernel, iterations=4)
        blur = cv2.GaussianBlur(erosion, (7, 7), 4)

        _, mask = cv2.threshold(
            blur, thresh=175, maxval=255, type=cv2.THRESH_BINARY)
        im_thresh_gray = cv2.bitwise_and(blur, mask)

        return im_thresh_gray

    def find_lines(self, im):
        im, _ = TimeReader.get_rotated(im, self.timer_templates[0])
        dst = cv2.Canny(im, 200, 200, None, 3)

        cv2.imshow('canny', dst)

        lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 50, 10)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(im, (l[0], l[1]), (l[2], l[3]),
                         (0, 0, 255), 1, cv2.LINE_AA)

        return im

    @staticmethod
    def convert_image_to_graph(image):
        """Builds a graphic by an image.
        The graphic shows count of pixels > 0 in grayscale.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            [int, int, ...]: The output graphic.
        """

        graph = []

        height, width = image.shape[:2]

        for x in range(width):
            counter = 0
            for y in range(height):
                if image[y, x] > 0:
                    counter += 1
            graph.append(counter)
            counter = 0

        return graph

    @staticmethod
    def get_rotated(image, template, err_height=config.DEFAULT_WRAP_POLAR_HEIGHT_ERROR):
        cut_image, _ = cf.ClockFace.search_face(image, template)
        rotated = cf.ClockFace.wrap_polar_face(
            cut_image, error_height=err_height)
        return cut_image, rotated
