from Layers.Layer import Layer
import numpy as np

class DenseLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward_prop(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward_prop(self, dY, learning_rate):
        dW = np.dot(dY, self.input.T)
        dX = np.dot(self.weights.T, dY) 
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dY
        return dX