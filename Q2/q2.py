# %% -- imports

# python libs
import os
import re
import random

# scientific libs
import cv2
import tensorflow as tf
import numpy as np

# %%

image_size = 200
path = 'data/data'


# %%

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


def process_and_get_data():
    X = []
    Y = []
    for root, dirs, files in os.walk(path):
        regex = '^{}/[0-9]+$'.format(path)
        if re.match(regex, root) is not None:
            label = int(root.split('/')[-1])
            for file in files:
                if '.jpg' in file:
                    directory = root + '/' + file

                    img = cv2.imread(directory)

                    if img is not None:
                        img = pad_image(img)
                        img = cv2.resize(img, (image_size, image_size))

                        x = np.reshape(img, image_size * image_size * 3)

                        y_true = np.zeros(7, dtype=float)
                        y_true[label - 1] = 1

                        X.append(x)
                        Y.append(y_true)

    return np.array(X), np.array(Y)


def shuffle(arrays, k=None):
    assert len(arrays) > 0
    length = len(arrays[0])
    if k is None:
        k = length
    mask = random.sample(range(length), k)

    return [array[mask] for array in arrays]


# %% -- load data

ds = process_and_get_data()
ds = shuffle(ds)

# %%

idx = 700

train_ds = ds[0][:idx], ds[1][:idx]
test_ds = ds[0][idx:], ds[1][idx:]


# %% funcs

# INIT WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


# INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


# CONV2D
def conv2d(x, W):
    # x --> [batch, H, W, Channels]
    # W --> [filter H, filter W, Channels In, Channels Out]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# POOLING
def max_pool_2by2(x):
    # x --> [batch, H, W, Channels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# CONV LAYER
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])  # ????
    return tf.nn.relu(conv2d(input_x, W) + b)


# Normal (fully connected)
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])  # ??
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# %% -- placeholders

X = tf.placeholder(name="x", dtype=tf.float32, shape=[None, image_size * image_size * 3])
y_true = tf.placeholder(name="y_true", dtype=tf.float32, shape=[None, 7])
x_image = tf.reshape(X, [-1, image_size, image_size, 3])

# %% build the network

convo1 = convolutional_layer(x_image, shape=[5, 5, 3, 8])
convo1_pooling = max_pool_2by2(convo1)

convo2 = convolutional_layer(convo1_pooling, shape=[5, 5, 8, 16])
convo2_pooling = max_pool_2by2(convo2)

convo2_flat = tf.reshape(convo2_pooling, shape=[-1, (image_size // 4) * (image_size // 4) * 16])
full_layer_one = tf.nn.relu(normal_full_layer(convo2_flat, 512))

y_pred = normal_full_layer(full_layer_one, 7)

# %% define the training

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# %% train

steps = 5000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):

        batch_x, batch_y = shuffle(train_ds, 50)

        sess.run(train, feed_dict={X: batch_x, y_true: batch_y})

        if i % 100 == 0:
            print("ON STEP: {}".format(i))
            print("ACCURACY: ")
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(accuracy, feed_dict={X: test_ds[0], y_true: test_ds[1]}))
            print('\n')
