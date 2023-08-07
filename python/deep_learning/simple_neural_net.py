import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.dz2 = output - y
        self.dW2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.W2.T) * sigmoid_derivative(self.a1)
        self.dW1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            self.update(lr)


if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(2, 5, 1)

    print("Before training:")
    print(nn.forward(X))

    nn.train(X, y, epochs=10000, lr=0.1)

    print("\nAfter training:")
    final_output = nn.forward(X)

    threshold = 0.5
    classified_output = (final_output >= threshold).astype(int)

    print("Final Output:")
    print(final_output)
    print("\nClassified Output:")
    print(classified_output)
