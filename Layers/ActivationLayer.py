from Layers.Layer import Layer
import numpy as np

class ActivationLayer(Layer):

    def __init__(self, activation, dActivation):
        self.activation = activation
        self.dActivation = dActivation

    def forward_prop(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward_prop(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.dActivation(self.input))