import numpy as np
from Client import WrappedClient
from Server import WrappedServer
import pandas as pd

dataset = './datasets/NJ2021.csv'  # the file containing client's data
m = 200                 # the domain size of values
n = 59869               # the number of clients
h = 50                  # the number of reports of each client
epsilon = 1             # the privacy parameter
mu = 0                  # the estimated mean


if __name__ == "__main__":
    numberOfBits = np.floor(np.log2(m)) + 1
    mean = np.zeros(h * n)
    data = pd.read_csv(dataset).to_numpy()
    server = WrappedServer(numberOfBits, epsilon)
    client = [WrappedClient(numberOfBits, epsilon) for _ in range(n)]

    for i in range(h):
        values = data[:, i]
        for j in range(n):
            mean[i * n + j] = values[j]
            [allV, allH] = client[j].report(values[j])
            for k in range(len(allV)):
                server.newValue(allV[k], allH[k], k)
        server.predicate()
    result = server.finish()
    for _, row in enumerate(result):
        for i, number in enumerate(row):
            mu += (number * n * 2 ** (len(row) - 1 - i)) / (n * h)
    consumed_budget = [client[i].consumedBudget() for i in range(n)]
    error = abs(np.mean(mean) - mu)

    print("epsilon=%f: Estimated Mean: %f  Real Mean: %f  Estimation Error: %f  Average Consumed Budget: %f"
          % (epsilon, mu, mean, abs(mu - mean), np.mean(consumed_budget)))
