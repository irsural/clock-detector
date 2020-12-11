import cv2
import unittest
import utility
import os

try:
    from detector import time_reader as tr
except Exception as _:
    utility.settings_sys_path()
    from detector import time_reader as tr

unittest.TestLoader.sortTestMethodsUsing = None

class TestReadTime(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestReadTime, self).__init__(*args, **kwargs)
        self.cache_images_answer = {}

    def test_00_readable(self):
        list_images = utility.get_list_images()

        for image in list_images:
            _image = cv2.imread(image)
            reader = tr.TimeReader(_image)
            time = reader.read_by_graph()

            self.assertIsNotNone(time)
            self.cache_images_answer[image] = time

    def test_01_correct_readable(self):
        for (hours, minutes, seconds), image in self.cache_images_answer:
            self.assertLessEqual(hours, 12)
            self.assertLessEqual(minutes, 60)
            self.assertLessEqual(seconds, 60)

if __name__ == "__main__":
    unittest.main()