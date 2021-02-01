import os
import sys

def settings_sys_path():
    need_dir = 'src'
    current_path = os.path.abspath(os.curdir)

    while os.path.basename(current_path) != need_dir:
        os.chdir('..')
        current_path = os.path.abspath(os.curdir)

    sys.path[0] = current_path

def get_list_images():
    list_images = []

    project_name = 'clock-detector'
    cache_name = 'cache'
    current_path = os.path.abspath(os.curdir)

    while os.path.basename(current_path) != project_name:
        os.chdir('..')
        current_path = os.path.abspath(os.curdir)

    current_path = os.path.join(current_path, cache_name)

    for directory in os.listdir(current_path):
        dir_path = os.path.join(current_path, directory)

        if os.path.isdir(dir_path):
            for image in os.listdir(dir_path):
                if image == '.DS_Store':
                    continue

                image_path = os.path.join(dir_path, image)
                list_images.append(os.path.abspath(image_path))

    return list_images
