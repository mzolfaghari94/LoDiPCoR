import csv
import numpy as np
from Client import Client
from Server import Server

m = 200                 # the domain size of values
n = 59869               # the number of clients
h = 50                  # the number of reports of each client
epsilon = 1             # privacy parameter
file = './datasets/NJ2021.csv'     # the file containing client's data
mu = 0                  # estimated mean
s = 5                   # segment size
gamma = 0.02            # privacy parameter for output peturbation
consumed_budget = []    # consumed budget of each client
mean = 0                # real mean

if __name__ == "__main__":
    server = Server(epsilon, m)  # create server
    with open(file) as file_obj:
        dataset = csv.reader(file_obj)
        next(dataset)
        for row in dataset:
            data = list(map(int, row))[:h]
            c = Client(epsilon, m, s, gamma)  # create client
            for x in data:
                mean += x / (n * h)     # calculate mean of values
                report = c.report(x)    # passing the data to the client and getting its report
                server.collect(report)  # passing the client's report to the server
            consumed_budget.append(c.budget)
    mu = server.estimation()
    print("epsilon=%f: Estimated Mean: %f  Real Mean: %f  Estimation Error: %f  Average Consumed Budget: %f"
          % (epsilon, mu, mean, abs(mu - mean), np.mean(consumed_budget)))
