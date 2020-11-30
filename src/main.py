import cv2

from detector import time_reader as tr

if __name__ == "__main__":
    image_path = '../cache/images/3.jpg'
    image = cv2.imread(image_path)

    reader = tr.TimeReader(image)
    hours, minutes, seconds = reader.read()

    print(f'{hours}:{minutes}:{seconds}')
