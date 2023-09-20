import Layers.DenseLayer as Dense
import Layers.ActivationLayer as Activation
import Activations.Tanh as Tanh
import Errors.Loss as loss
import numpy as np

network = [
    Dense.DenseLayer(2,3),
    Tanh.Tanh(),
    Dense.DenseLayer(3,1),
    Tanh.Tanh()
]

epochs = 10000
learning_rate = 0.1

X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
Y = np.reshape([0, 1, 1, 0], (4,1,1))

for i in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        # forward propagation
        output = x
        for layer in network:
            output = layer.forward_prop(output)
        
        # calculate error
        error += loss.mse(y, output)

        # backward propagation
        output_gradient = loss.mse_derivative(y, output)
        for layer in network[::-1]:
            output_gradient = layer.backward_prop(output_gradient, learning_rate)
    
    error /= len(X)
    print("Error for epoch " + str(i + 1) + ": " + str(error))
