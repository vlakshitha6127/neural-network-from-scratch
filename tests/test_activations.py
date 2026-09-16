import numpy as np
from src.activations import relu, relu_derivative


def test_relu():
    x = np.array([-3, -1, 0, 2, 5])

    result = relu(x)

    expected = np.array([0, 0, 0, 2, 5])

    assert np.array_equal(result, expected)


def test_relu_derivative():
    x = np.array([-3, -1, 0, 2, 5])

    result = relu_derivative(x)

    expected = np.array([0, 0, 0, 1, 1])

    assert np.array_equal(result, expected)