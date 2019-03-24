# %% -- imports

import tensorflow as tf

# %% -- load data

from read_data import get_data

X_d, Y_d = get_data()


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

X = tf.placeholder(name="x", dtype=tf.float32, shape=[None, 40000])
y_true = tf.placeholder(name="y_true", dtype=tf.float32, shape=[None, 7])
x_image = tf.reshape(X, [-1, 200, 200, 1])

# %% build the network

convo1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
convo1_pooling = max_pool_2by2(convo1)

convo2 = convolutional_layer(convo1_pooling, shape=[5, 5, 32, 64])
convo2_pooling = max_pool_2by2(convo2)

convo2_flat = tf.reshape(convo2_pooling, shape=[-1, 50 * 50 * 64])
full_layer_one = tf.nn.relu(normal_full_layer(convo2_flat, 1024))

y_pred = normal_full_layer(full_layer_one, 7)

# %% define the training

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

# %% train

steps = 500
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):

        batch_x = X_d[i:i + 50]
        batch_y = Y_d[i:i + 50]

        sess.run(train, feed_dict={X: batch_x, y_true: batch_y})

        if i % 100 == 0:
            print("ON STEP: {}".format(i))
            print("ACCURACY: ")
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
            print(sess.run(accuracy, feed_dict={X: batch_x, y_true: batch_y}))
            print('\n')

# %%
