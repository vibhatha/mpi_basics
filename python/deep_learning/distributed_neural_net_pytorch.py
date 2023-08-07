import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool, cpu_count


# Define the Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


def train_on_batch(args):
    model, X, y, epochs = args
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.state_dict()  # Return updated model parameters


def parallel_training():
    model = NeuralNetwork(2, 5, 1)

    # Set up data for XOR problem
    X = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.Tensor([[0], [1], [1], [0]])

    # Splitting data into batches (assuming we have a lot more data)
    print(f"Parallelism : {2}")
    batch_size = len(X) // 2
    X_batches = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]
    y_batches = [y[i : i + batch_size] for i in range(0, len(y), batch_size)]

    # Using Python's multiprocessing Pool
    with Pool(2) as p:
        params = [(model, X_batches[i], y_batches[i], 10000) for i in range(2)]
        results = p.map(train_on_batch, params)

    # Averaging model parameters for simplicity (more advanced strategies can be used)
    avg_state_dict = {}
    for key in results[0].keys():
        avg_state_dict[key] = sum([res[key] for res in results]) / len(results)

    model.load_state_dict(avg_state_dict)

    # Test the model
    with torch.no_grad():
        test_output = model(X)
        classified_output = (test_output > 0.5).float()
        print("Final Output:", test_output)
        print("Classified Output:", classified_output)


if __name__ == "__main__":
    parallel_training()
