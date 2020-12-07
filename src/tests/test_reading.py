import cv2
import unittest
import utility

try:
    from detector import time_reader as tr
except Exception as _:
    utility.settings_sys_path()
    from detector import time_reader as tr

class TestReadTime(unittest.TestCase):

    def test_readable(self):
        list_images = utility.get_list_images()

        for image in list_images:
            _image = cv2.imread(image)
            reader = tr.TimeReader(_image)

            self.assertIsNotNone(reader.read())

if __name__ == "__main__":
    unittest.main()