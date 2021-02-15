import copy
import cv2
import numpy as np
import config


class ClockFace:
    """This class is used for computing working with a clock face.
    """

    @staticmethod
    def search_face(im1, im2):
        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.15

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(
            im1, keypoints1, im2, keypoints2, matches, None)
#         plt.imshow(imMatches), plt.show()

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h

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

        # TODO: There is gonna be something to wrap polar a part of image
        polarImage = cv2.warpPolar(rotate, (height, width), (centre, centre),
                                   centre, cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)

        cropImage = copy.deepcopy(polarImage[0:width, error_height:height])
        cropImage = cv2.rotate(cropImage, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return cropImage

    @staticmethod
    def wrap_polar_image(image,
                         width=config.DEFAULT_WRAP_POLAR_WIDTH,
                         height=config.DEFAULT_WRAP_POLAR_HEIGHT,
                         error_height=config.DEFAULT_WRAP_POLAR_HEIGHT_ERROR):
        rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        centre = (0, 0)

        polarImage = cv2.warpPolar(image, (height, width), centre,
                                   image.shape[0], cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)

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
