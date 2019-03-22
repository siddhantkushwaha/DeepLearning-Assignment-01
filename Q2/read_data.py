import os
import random
import re
import cv2


def pad_image(img, size=(500, 500)):
    diff_x = size[1] - img.shape[1]
    diff_y = size[0] - img.shape[0]

    if diff_x < 0 or diff_y < 0:
        raise Exception("Invalid image size for padding.")

    left_padding = diff_x // 2
    right_padding = diff_x - left_padding

    top_padding = diff_y // 2
    bottom_padding = diff_y - top_padding

    return cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT)


def get_data():
    data = []
    for root, dirs, files in os.walk('data/data'):
        if re.match('^data/data/[0-9]+$', root) is not None:
            label = int(root.split('/')[2])
            for file in files:
                directory = root + '/' + file
                img = cv2.imread(directory)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = pad_image(img)
                data.append((img, label, directory))
    return data


def get_train_test_data(train_ratio=0.7, data=None):
    if data is None:
        data = get_data()
    size = len(data)
    random.shuffle(data)

    slice_idx = int(size * train_ratio)

    return data[:slice_idx], data[slice_idx:]
