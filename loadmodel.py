import numpy as np
import matplotlib.pyplot as plt
from model import Convolution, MaxPool, NeuralNetwork, arg_max

def display_logo():
    print("------------------------------------------------------------------")
    print("|               NEURAL NETWORK, WHICH DETERMINE DIGIT             |")
    print("------------------------------------------------------------------")

def load_model_and_predict(image_path):
    display_logo()
    print("You are loading an image, and this neural network will determine the digit.")
   
    conv = Convolution()
    max_pool = MaxPool()
    neural_net = NeuralNetwork()

    img = plt.imread(image_path)
    if img.ndim == 3:
        img = img.mean(axis=2)  # RGB to Grayscale

    img = img / 255.0

    conv_output = conv.forward(img)
    max_pool_output = max_pool.forward(conv_output)
    output = neural_net.feedforward(max_pool_output, np.zeros(10))  

    predicted_digit = arg_max(output)
    print(f'Predicted digit: {predicted_digit}')

if __name__ == "__main__":
    predict = input("Enter the file path of the image: ")  # input 
    load_model_and_predict(predict)
