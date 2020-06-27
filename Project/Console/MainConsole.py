from Project.AI.Sigmoid import *

# Test
new_inputs = np.array([1,1,0])
output = sigmoid(np.dot(new_inputs,synaptic_weights))

print("New case:")
print(output)

# New tests
while True:
    print("Inser new valuse (n to exit): ")
    data = input()

    if data == "n":
        break

    output = evaluate(data)

    print("Excpected result: ")
    print(output)

print("App finished")
