import numpy as np
import csv
from Server import Server
from Client import Client

m = 200                 # the domain size of values
n = 59869               # the number of clients
h = 50                  # the number of reports of each client
epsilon_1 = 1           # privacy parameter for single report
epsilon_inf = 2         # privacy parameter for infinity reports
file = './datasets/NJ2021.csv'     # the file containing client's data
mu = 0                  # estimated mean
real = []               # all values of clients
reports = []            # all sanitized values sent to server
consumed_budget = []                 # consumed budget of clients
mean = 0                # real mean of all values


if __name__ == "__main__":
    with open(file) as file_obj:
        dataset = csv.reader(file_obj)
        next(dataset)
        for row in dataset:
            data = list(map(int, row))[:h]
            client = Client(epsilon_inf, epsilon_1, m)
            real.append(data)
            reports.append(client.report(data))
            consumed_budget.append(client.budget)

    server = Server(epsilon_inf, epsilon_1)
    est_freq = server.estimate(np.concatenate(reports, axis=0))
    real_freq = np.zeros(m)
    values, freq = np.unique(np.concatenate(real, axis=0), return_counts=True)
    for i in range(len(values)):
        real_freq[values[i]] = freq[i] / (n * h)

    for i in range(m):
        mu += est_freq[i] * (i + 1)
        mean += real_freq[i] * (i + 1)

    print("epsilon1=%f: Estimated Mean: %f  Real Mean: %f  Estimation Error: %f  Average Consumed Budget: %f"
          % (epsilon_1, mu, mean, abs(mu - mean), np.mean(consumed_budget)))
