import cv2
import numpy as np


from detector import time_reader as tr

def read_video(is_clock):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while(True):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)

        try:
            reader = tr.TimeReader(frame)
            answer = reader.read_clock() if is_clock else reader.read_timer()
            for idx, item in enumerate(answer):
                item = item if item > 9 else f'0{item}'

                if idx < len(answer) - 1:
                    print(item, end=':')
                else:
                    print(item)
        except Exception as ex:
            print('Cannot read time!')

        # key = cv2.waitKey(1)

        # if key == ord('q'):
        #     reader = tr.TimeReader(frame)

        #     try:
        #         answer = reader.read_clock() if is_clock else reader.read_timer()

        #         for idx, item in enumerate(answer):
        #             item = item if item > 9 else f'0{item}'

        #             if idx < len(answer) - 1:
        #                 print(item, end=':')
        #             else:
        #                 print(item)

        #     except Exception as ex:
        #         print('Cannot read time!')
        if cv2.waitKey(1) == ord('e'):
            vid.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    read_video(False)
    # # image_path = '../cache/exceptions/4.jpg'
    # # image_path = '../cache/clock/4.jpg'
    # image_path = '../cache/timer/4.jpg'
    # image = cv2.imread(image_path)

    # is_clock = False
    # reader = tr.TimeReader(image)
    # answer = reader.read_clock() if is_clock else reader.read_timer()

    # for idx, item in enumerate(answer):
    #     item = item if item > 9 else f'0{item}'

    #     if idx < len(answer) - 1:
    #         print(item, end=':')
    #     else:
    #         print(item)