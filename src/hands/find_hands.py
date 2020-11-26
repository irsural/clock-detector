import cv2
import numpy as np

def find_contours(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    blur1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    blur2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    diff_frame = cv2.absdiff(blur1, blur2)
    thresh_frame = cv2.threshold(
        diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh_frame = cv2.erode(thresh_frame, kernel, iterations=3)

    # cv2.imshow('f', diff_frame)
    # cv2.imshow('g', thresh_frame)

    cnts, hier = cv2.findContours(
        thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    return cnts

def get_hands(contours):
    hands = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000 or area < 1000:
            continue

        # radius = 0.1
        # accuracy = radius * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, accuracy, True)

        # x, y, width, height = cv2.boundingRect(approx)

        x, y, width, height = cv2.boundingRect(contour)

        # if len(approx) == 2:
        roi = img1[y:y+height, x:x+width]

        cv2.imshow('roi', roi)
        if cv2.waitKey() == ord('q'):
            hands.append(roi)
            # cv2.imwrite(f'hands/{idx}.jpg', roi)

    return hands

if __name__ == "__main__":
    img1 = cv2.imread('cache/images/2.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('cache/images/1.jpg', cv2.IMREAD_COLOR)

    cnts = find_contours(img1, img2)

    if cnts is not None:
        list_hands = get_hands(cnts)

        for idx, img in enumerate(list_hands):
            cv2.imwrite(f'cache/template/{idx+1}.jpg', img)
