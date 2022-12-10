import numpy as np
import csv
from Client import Client
from Server import Server


m = 200                 # the values domain size
n = 59869               # the number of clients
h = 50                  # the number of reports of each client
thresh = 5              # the threshold of each client
coef = 10               # the coefficient of budget of each client
epsilon_1 = 1           # privacy parameter for single report
epsilon_inf = 2         # privacy parameter for infinity reports
dataset = 'NJ2021.csv'  # the file containing client's data
real = []               # all values of clients
reports = []            # all sanitized values sent to server
cb = []                 # consumed budget of clients
mean = 0                # real mean of all values
freq = np.zeros(m)      # number of repeats of values


if __name__ == "__main__":
    with open(dataset) as file_obj:
        data = csv.reader(file_obj)
        next(data)
        for row in data:
            values = list(map(int, row))[:h]
            budget = coef * epsilon_inf
            client = Client(epsilon_inf, epsilon_1, m, thresh, budget)
            real.append(values)
            reports.append(client.report(values))
            cb.append(budget - client.budget)
    server = Server(epsilon_inf, epsilon_1)
    mu = server.estimate(np.concatenate(reports, axis=0))
    values, freq = np.unique(np.concatenate(real, axis=0), return_counts=True)
    for i in range(len(values)):
        mean += values[i] * freq[i] / (n * h)
    print("epsilon1=%f: Estimated Mean: %f  Real Mean: %f  Estimation Error: %f  Average Consumed Budget: %f"
          % (epsilon_1, mu, mean, abs(mu - mean), np.mean(cb)))
