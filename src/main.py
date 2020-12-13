import cv2

from detector import time_reader as tr

if __name__ == "__main__":
    # image_path = '../cache/exceptions/4.jpg'
    # image_path = '../cache/clock/4.jpg'
    image_path = '../cache/timer/4.jpg'
    image = cv2.imread(image_path)

    is_clock = False
    reader = tr.TimeReader(image)
    answer = reader.read_clock() if is_clock else reader.read_timer()

    for idx, item in enumerate(answer):
        item = item if item > 9 else f'0{item}'
        
        if idx < len(answer) - 1:
            print(item, end=':')
        else:
            print(item)