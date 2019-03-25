import os
import re
import cv2
import numpy as np

size = 300


def pad_image(img):
    diff_x = 500 - img.shape[1]
    diff_y = 500 - img.shape[0]

    if diff_x < 0 or diff_y < 0:
        raise Exception("Invalid image size for padding.")

    left_padding = diff_x // 2
    right_padding = diff_x - left_padding

    top_padding = diff_y // 2
    bottom_padding = diff_y - top_padding

    return cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT)


def process_data():
    for root, dirs, files in os.walk('data/data'):
        if re.match('^data/data/[0-9]+$', root) is not None:
            label = int(root.split('/')[2])
            for file in files:
                directory = root + '/' + file

                img = cv2.imread(directory)

                if img is not None:
                    img = pad_image(img)

                    os.system('mkdir -p processed_data/data/%d' % label)
                    cv2.imwrite('processed_data/data/%d/%s' % (label, file), cv2.resize(img, (size, size)))


def get_data():
    row = []
    for root, dirs, files in os.walk('processed_data/data'):
        if re.match('^processed_data/data/[0-9]+$', root) is not None:
            label = int(root.split('/')[2])
            for file in files:
                directory = root + '/' + file

                img = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    x = np.reshape(img, size * size)

                    y_true = np.zeros(7, dtype=float)
                    y_true[label - 1] = 1

                    row.append(np.ndarray.tolist(np.concatenate((x, y_true))))

    return np.array(row)


# %%--

process_data()
ds = get_data()
