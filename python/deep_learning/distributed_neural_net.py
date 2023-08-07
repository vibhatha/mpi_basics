from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    # For this example, I'm still using XOR but in a real-world scenario you'd use a larger dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Splitting dataset across processors
    num_samples_per_processor = X.shape[0] // size
    X_local = X[
        rank * num_samples_per_processor : (rank + 1) * num_samples_per_processor
    ]
    y_local = y[
        rank * num_samples_per_processor : (rank + 1) * num_samples_per_processor
    ]

    nn = NeuralNetwork(2, 5, 1)

    # Only root prints initial results
    if rank == 0:
        print("Before training:")
        print(nn.forward(X))

    for epoch in range(10000):
        local_output = nn.forward(X_local)
        nn.backward(X_local, y_local, local_output)

        # Gathering gradients from all processes
        dW1_global = np.zeros_like(nn.dW1)
        dW2_global = np.zeros_like(nn.dW2)
        db1_global = np.zeros_like(nn.db1)
        db2_global = np.zeros_like(nn.db2)

        comm.Allreduce([nn.dW1, MPI.DOUBLE], [dW1_global, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([nn.dW2, MPI.DOUBLE], [dW2_global, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([nn.db1, MPI.DOUBLE], [db1_global, MPI.DOUBLE], op=MPI.SUM)
        comm.Allreduce([nn.db2, MPI.DOUBLE], [db2_global, MPI.DOUBLE], op=MPI.SUM)

        # Averaging gradients
        nn.dW1 = dW1_global / size
        nn.dW2 = dW2_global / size
        nn.db1 = db1_global / size
        nn.db2 = db2_global / size

        nn.update(0.1)

    # Only root prints final results
    if rank == 0:
        print("\nAfter training:")
        final_output = nn.forward(X)

        threshold = 0.5
        classified_output = (final_output >= threshold).astype(int)

        print("Final Output:")
        print(final_output)
        print("\nClassified Output:")
        print(classified_output)
