import cv2

from detector import time_reader as tr

if __name__ == "__main__":
    image_path = '../cache/exceptions/1.jpg'
    # image_path = '../cache/clock/4.jpg'
    # image_path = '../cache/timer/1.jpg'
    image = cv2.imread(image_path)

    is_clock = True
    reader = tr.TimeReader(image, is_clock)

    if is_clock:
        hours, minutes, seconds = reader.read_by_graph()

        hours = hours if hours > 9 else f'0{hours}'
        minutes = minutes if minutes > 9 else f'0{minutes}'
        seconds = seconds if seconds > 9 else f'0{seconds}'

        print(f'{hours}:{minutes}:{seconds}')
    else:
        seconds = reader.read()
        print(f'{seconds}')