from Layers.ActivationLayer import ActivationLayer
import numpy as np

class Sigmoid(ActivationLayer):

    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        dSigmoid = lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        super().__init__(sigmoid, dSigmoid)