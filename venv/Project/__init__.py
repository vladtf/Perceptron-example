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

print("Random initial weights:")
print(synaptic_weights)

for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustmets = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustmets

print("Weights after learning:")
print(synaptic_weights)

print("Results:")
print(outputs)

# # Test
# new_inputs = np.array([1,1,0])
# output = sigmoid(np.dot(new_inputs,synaptic_weights))
#
# print("New case:")
# print(output)

# New tests
while True:
    print("Inser new valuse (n to exit): ")
    data = input()

    if data == "n":
        break

    tokens = list(map(int, data.split(" ")))
    new_inputs = np.array([tokens[0], tokens[1], tokens[2]])
    output = sigmoid(np.dot(new_inputs, synaptic_weights))

    print("Excpected result: ")
    print(output)

print("App finished")
