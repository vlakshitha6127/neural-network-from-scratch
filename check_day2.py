'''import numpy as np

from src.activations import relu, relu_derivative
from src.activations import sigmoid, sigmoid_derivative
from src.activations import softmax
from src.losses import mse, cross_entropy


print("========== RELU ==========")

x = np.array([-3, -1, 0, 2, 5])

print("Input:", x)
print("ReLU:", relu(x))
print("ReLU derivative:", relu_derivative(x))


print("\n========== SIGMOID ==========")

x = np.array([-10, 0, 10])

print("Input:", x)
print("Sigmoid:", sigmoid(x))
print("Sigmoid derivative:", sigmoid_derivative(x))


print("\n========== SOFTMAX ==========")

logits = np.array([[3, 1, 0]])

probabilities = softmax(logits)

print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum:", np.sum(probabilities))


print("\n========== MSE ==========")

y_true = np.array([1, 0])
y_pred = np.array([0.8, 0.3])

print("True:", y_true)
print("Predicted:", y_pred)
print("MSE:", mse(y_true, y_pred))


print("\n========== CROSS ENTROPY ==========")

y_true = np.array([[1, 0, 0]])
logits = np.array([[3, 1, 0]])

print("True class: 0")
print("Logits:", logits)
print("Cross entropy:", cross_entropy(y_true, logits))'''

import numpy as np
from src.layers import Dense


print("========== DENSE LAYER ==========")

np.random.seed(42)

layer = Dense(4, 3)

X = np.random.randn(5, 4)

output = layer.forward(X)

print("Weights shape:", layer.weights.shape)
print("Bias shape:", layer.bias.shape)
print("Input shape:", X.shape)
print("Output shape:", output.shape)

print("\nOutput:")
print(output)


print("\n========== HE INITIALIZATION ==========")

layer_he = Dense(100, 50, initialization="he")

print("Expected standard deviation:", np.sqrt(2 / 100))
print("Actual standard deviation:", np.std(layer_he.weights))


print("\n========== XAVIER INITIALIZATION ==========")

layer_xavier = Dense(100, 50, initialization="xavier")

print("Expected standard deviation:", np.sqrt(1 / 100))
print("Actual standard deviation:", np.std(layer_xavier.weights))


print("\n========== FINITE VALUES ==========")

print("Output contains NaN:",
      np.any(np.isnan(output)))

print("Output contains infinity:",
      np.any(np.isinf(output)))