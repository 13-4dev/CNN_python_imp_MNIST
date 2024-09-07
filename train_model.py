import tensorflow as tf
import numpy as np
import os


os.makedirs('weights', exist_ok=True)

# MNIST 4 TensorFlow
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def mse_loss(X, Y):
    size_output = len(X)
    sum_squared = np.sum((np.array(X) - np.array(Y)) ** 2)
    return sum_squared / (2 * size_output)


class Convolution:
    def __init__(self):
        self.conv_layer = np.zeros((5, 28, 28))
        self.sig_layer = np.zeros((5, 28, 28))
        self.kernels = np.random.uniform(-1, 1, (5, 3, 3))

    def forward(self, input_layer):
        for filter_dim in range(5):
            for i in range(28):
                for j in range(28):
                    conv_sum = 0
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if 0 <= i + k < 28 and 0 <= j + l < 28:
                                conv_sum += input_layer[i + k, j + l] * self.kernels[filter_dim, k + 1, l + 1]
                    self.conv_layer[filter_dim, i, j] = conv_sum
        self.sig_layer = sigmoid(self.conv_layer)
        return self.sig_layer

    def save_weights(self, filepath):
        np.save(filepath, self.kernels)

#  MaxPool
class MaxPool:
    def __init__(self):
        self.max_pooling = np.zeros((5, 28, 28))
        self.max_layer = np.zeros((5, 14, 14))

    def forward(self, sig_layer):
        for filter_dim in range(5):
            for i in range(0, 28, 2):
                for j in range(0, 28, 2):
                    max_i, max_j = i, j
                    cur_max = sig_layer[filter_dim, i, j]
                    for k in range(2):
                        for l in range(2):
                            if sig_layer[filter_dim, i + k, j + l] > cur_max:
                                max_i = i + k
                                max_j = j + l
                                cur_max = sig_layer[filter_dim, max_i, max_j]
                    self.max_pooling[filter_dim, max_i, max_j] = 1
                    self.max_layer[filter_dim, i // 2, j // 2] = cur_max
        return self.max_layer

#  NeuralNetwork
class NeuralNetwork:
    def __init__(self):
        self.loss = 0
        self.dense_input = np.zeros(980)
        self.dense_sum = np.zeros(120)
        self.dense_sigmoid = np.zeros(120)
        self.dense_sum2 = np.zeros(10)
        self.dense_softmax = np.zeros(10)

        self.dense_w = np.random.uniform(-1, 1, (980, 120))
        self.dense_b = np.random.uniform(-1, 1, 120)
        self.dense_w2 = np.random.uniform(-1, 1, (120, 10))
        self.dense_b2 = np.random.uniform(-1, 1, 10)

    def feedforward(self, max_layer, labelY):
        k = 0
        for filter_dim in range(5):
            for i in range(14):
                for j in range(14):
                    self.dense_input[k] = max_layer[filter_dim][i][j]
                    k += 1

        self.dense_sum = np.dot(self.dense_input, self.dense_w) + self.dense_b
        self.dense_sigmoid = sigmoid(self.dense_sum)

        self.dense_sum2 = np.dot(self.dense_sigmoid, self.dense_w2) + self.dense_b2

        den = np.sum(np.exp(self.dense_sum2))
        self.dense_softmax = np.exp(self.dense_sum2) / den

        self.loss = mse_loss(self.dense_softmax, labelY)
        return self.dense_softmax

    def backpropagation(self, labelX, labelY):
        delta4 = labelX - labelY

        self.dw2 = np.outer(self.dense_sigmoid, delta4)

        delta3 = np.dot(self.dense_w2, delta4) * d_sigmoid(self.dense_sum)

        self.dw1 = np.outer(self.dense_input, delta3)

        return delta3

    def update_weights(self, learning_rate):
        self.dense_b -= learning_rate * self.dw1.sum(axis=0)
        self.dense_b2 -= learning_rate * self.dw2.sum(axis=0)
        self.dense_w2 -= learning_rate * self.dw2
        self.dense_w -= learning_rate * self.dw1

    def save_weights(self, filepath):
        np.savez(filepath, dense_w=self.dense_w, dense_b=self.dense_b, dense_w2=self.dense_w2, dense_b2=self.dense_b2)


def train_network():
    x_train, y_train, x_test, y_test = load_mnist_data()

    conv = Convolution()
    max_pool = MaxPool()
    neural_net = NeuralNetwork()

    learning_rate = 0.01
    epochs = 10

    for epoch in range(epochs):
        for i in range(len(x_train)):
            input_image = x_train[i]
            label = y_train[i]

            conv_output = conv.forward(input_image)
            max_pool_output = max_pool.forward(conv_output)
            output = neural_net.feedforward(max_pool_output, label)

            prev_dense = neural_net.backpropagation(output, label)

            neural_net.update_weights(learning_rate)

            if i % 1000 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Step {i}, Loss: {neural_net.loss}')

    conv.save_weights('weights/conv_weights.npy')
    neural_net.save_weights('weights/neural_net_weights.npz')

    test_accuracy = evaluate_network(neural_net, conv, max_pool, x_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

def evaluate_network(neural_net, conv, max_pool, x_test, y_test):
    correct_predictions = 0
    for i in range(len(x_test)):
        input_image = x_test[i]
        label = y_test[i]

        conv_output = conv.forward(input_image)
        max_pool_output = max_pool.forward(conv_output)
        output = neural_net.feedforward(max_pool_output, label)

        if np.argmax(output) == np.argmax(label):
            correct_predictions += 1

    accuracy = (correct_predictions / len(x_test)) * 100
    return accuracy

if __name__ == "__main__":
    train_network()
