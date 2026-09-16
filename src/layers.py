import numpy as np


class Dense:
    def __init__(self, input_size, output_size, initialization="he"):
        self.input_size = input_size
        self.output_size = output_size

        if initialization == "he":
            self.weights = (
                np.random.randn(input_size, output_size)
                * np.sqrt(2 / input_size)
            )

        elif initialization == "xavier":
            self.weights = (
                np.random.randn(input_size, output_size)
                * np.sqrt(1 / input_size)
            )

        else:
            raise ValueError("Unknown initialization")

        self.bias = np.zeros(output_size)

    def forward(self, X):
        return X @ self.weights + self.bias