import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))




training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

# print("Random initial weights:")
# print(synaptic_weights)


# print("Weights after learning:")
# print(synaptic_weights)
#
# print("Results:")
# print(outputs)


for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustmets = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustmets


def evaluate(a,b,c):
    new_inputs = np.array([a, b, c])
    output = round(float(sigmoid(np.dot(new_inputs, synaptic_weights))))
    return output