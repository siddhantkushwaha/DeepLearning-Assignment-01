# %% all important libraries

import numpy as np
import matplotlib.pyplot as plt

# %%: declaring the data variables

data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

X = data[:, :-1]
X = np.insert(X, 0, 1, axis=1)

Y_true = data[:, len(data) - 2:]

# %% initializing the weights randomly

W1 = np.random.rand(2, 3)
W2 = np.random.rand(1, 3)


# %% defining the activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff_sigmoid(x):
    temp = sigmoid(x)
    return temp * (1 - temp)


# %% defining the necessary functions for feed_forward and back_propagation

def feed_forward(x):
    net1 = np.matmul(W1, x)
    x1 = sigmoid(net1)
    x1 = np.insert(x1, 0, 1, axis=0)

    net2 = np.matmul(W2, x1)
    y = sigmoid(net2)

    return net1, x1, net2, y


def backprop(x, y_true, net1, x1, net2, y):
    del2 = (y - y_true) * diff_sigmoid(net2)
    Delta_W2 = np.matmul(del2, np.transpose(x1))

    del1 = np.matmul(np.transpose(np.delete(W2, 0, axis=1)), del2) * diff_sigmoid(net1)
    Delta_W1 = np.matmul(del1, np.transpose(x))

    return Delta_W1, Delta_W2


def get_gradients(x, y_true):
    net1, x1, net2, y = feed_forward(x)

    Delta_W1, Delta_W2 = backprop(x, y_true, net1, x1, net2, y)

    return Delta_W1, Delta_W2


# %% some utility functions to monitor and analyze the training

def display_results():
    _, _, _, Y = feed_forward(np.transpose(X))

    print('Predicted Values - ', Y[0])

    error = np.average(np.square(Y_true - np.transpose(Y)), axis=0)[0]
    print('Error - ', error, end='\n\n')

    return error


_x = np.arange(-2, 3, step=0.1)
_y = np.arange(-2, 3, step=0.1)
xx, yy = np.meshgrid(_x, _y)
_z = np.zeros(xx.shape)
plt.ion()


def plot():
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            inp = np.array([1, xx[i][j], yy[i][j]])
            _, _, _, out = feed_forward(inp)
            _z[i][j] = out[0]

    plt.clf()
    plt.contourf(xx, yy, _z, cmap='RdGy')
    plt.colorbar()
    plt.draw()


# %% the functions that defines the training of the perceptron

def train(epochs, learning_rate, threshold):
    global W1, W2

    for epoch in range(epochs):
        for i in range(len(data)):
            x = np.reshape(X[i], [len(X[i]), 1])
            y_true = Y_true[i]

            Delta_W1, Delta_W2 = get_gradients(x, y_true)

            W1 = W1 - learning_rate * Delta_W1
            W2 = W2 - learning_rate * Delta_W2

        # uncomment the below two lines to see the plot
        # if 0 == epoch % 100:
        #     plot()

        if epoch % (epochs / 100) == 0:
            print('Epoch Number - ', epoch + 1)
            error = display_results()

            if error < threshold:
                print('Error is less than threshold, threshold -> ' + str(threshold) + ' Error -> ' + str(error))
                break


# %% call the function to train the network

train(epochs=10000, learning_rate=0.4, threshold=0.001)
