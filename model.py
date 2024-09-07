import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def arg_max(vec):
    return int(np.argmax(vec))

#  Convolution
class Convolution:
    def __init__(self):
        self.conv_layer = np.zeros((5, 28, 28))
        self.sig_layer = np.zeros((5, 28, 28))
        self.kernels = np.load('D:\\data\\test\\AI_STUFF\\CNN_python_imp_MNIST\\weights\\conv_weights.npy')

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
        weights = np.load('D:\\data\\test\\AI_STUFF\\CNN_python_imp_MNIST\\weights\\neural_net_weights.npz')
        self.dense_w = weights['dense_w']
        self.dense_b = weights['dense_b']
        self.dense_w2 = weights['dense_w2']
        self.dense_b2 = weights['dense_b2']

        self.dense_input = np.zeros(980)
        self.dense_sum = np.zeros(120)
        self.dense_sigmoid = np.zeros(120)
        self.dense_sum2 = np.zeros(10)
        self.dense_softmax = np.zeros(10)

    def feedforward(self, max_layer, _):
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

        return self.dense_softmax
