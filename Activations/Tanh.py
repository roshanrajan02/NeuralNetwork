from Layers.ActivationLayer import ActivationLayer
import numpy as np

class Tanh(ActivationLayer):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        dTanh = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, dTanh)