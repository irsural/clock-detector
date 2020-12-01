import cv2

from detector import time_reader as tr

if __name__ == "__main__":
    image_path = '../cache/exceptions/2.jpg'
    # image_path = '../cache/images/4.jpg'
    image = cv2.imread(image_path)

    reader = tr.TimeReader(image)
    hours, minutes, seconds = reader.read()

    hours = hours if hours > 9 else f'0{hours}' 
    minutes = minutes if minutes > 9 else f'0{minutes}'
    seconds = seconds if seconds > 9 else f'0{seconds}'

    print(f'{hours}:{minutes}:{seconds}')
