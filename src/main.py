import cv2
import config

from detector import clock_face as cf
from detector import detector_template as dt


def detect_sift(template, image):
    detector = dt.DetectorTemplate(template)
    detector.detectBySIFT(image, show_detected=True)


def detect_without_mask(template, image):
    detector = dt.DetectorTemplate(template)
    val, loc, region = detector.detectWithoutMask(image, show_detected=True)


def detect_rotate(template, image, face):
    detector = dt.DetectorTemplate(template)

    len_contours, val, loc, region = detector.detect(
        image.copy(), face.radius, face.center)

    copy_origin_image = region.copy()
    gray = cv2.cvtColor(copy_origin_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 170, 220, 0)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_TC89_KCOS)
    cnts = imutils.grab_contours(cnts)

    maskROI = None
    x, y = None, None

    # ensure at least one contour was found
    if len(cnts) > 0:
        # grab the largest contour, then draw a mask for the pill
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)

        # compute its bounding box of pill, then extract the ROI,
        # and apply the mask
        (x, y, w, h) = cv2.boundingRect(c)
        imageROI = region[y:y + h, x:x + w]
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI,
                                   mask=maskROI)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 15
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(image) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(maskROI, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        if lines is not None:
            max_line = lines[0]
            for x1, y1, x2, y2 in max_line:
                max_dist = getDistanceBetweenPoints((x1, y1), (x2, y2))

            for line in lines:
                for x1, y1, x2, y2 in line:
                    dist = getDistanceBetweenPoints((x1, y1), (x2, y2))
                    if dist > max_dist:
                        max_line = line
                        max_dist = dist

        #             cv2.line(imageROI, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #             cv2.imshow('g', imageROI)
        #             cv2.waitKey()

    dx = loc[0]
    dy = loc[1]

    for x1, y1, x2, y2 in max_line:
        x1 += dx + x
        x2 += dx + x
        y1 += dy + y
        y2 += dy + y

        copy_image = image.copy()

        cv2.line(copy_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('g', copy_image)
        cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    template_path = 'cache/template/1.jpg'
    template = cv2.imread(template_path)

    image_path = 'cache/images/1.jpg'
    image = cv2.imread(image_path)

    face = cf.ClockFace(image)
    cutImage, centre, radius = face.computeClockFace()

    cv2.circle(cutImage, centre, 1, (0, 0, 255), 2)
    cv2.imshow('image', cutImage)
    cv2.waitKey()

    # detect_without_mask(template, image)
    # detect_rotate(template, image, face)
