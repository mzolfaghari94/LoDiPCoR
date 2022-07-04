import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

epsilon_1 = 1  # privacy budget in each instantaneous randomized response
epsilon_inf = 2  # privacy budget in each permanent randomized response
clients = int(1e6)  # number of clients
m = 10  # client's counter values are integers in [0, m)
k = 10


def plot(real_freq, est_freq, rng):  # plot the real and estimated histogram
    x_axis = np.arange(rng)
    plt.bar(x_axis - 0.25, real_freq, label='Real Freq', width=0.5)
    plt.bar(x_axis + 0.25, est_freq, label='Est Freq', width=0.5)
    plt.ylabel('Normalized Frequency')
    plt.xlabel('Domain Values')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.show()


if __name__ == "__main__":
    part = 0.6
    MAE = []
    MSE = []
    avg_err = []
    con_b = []
    thresh = 0.5
    error = []  # estimating error for each value
    consumed_budget = []  # consumed budget of each client
    responses = []  # number of responses of each client
    real = np.zeros(k)  # real histogram
    estimate = np.zeros(k)  # real histogram
    s = Server(epsilon_inf * part, epsilon_1, m)  # create the server
    avg = 0

    for _ in range(clients):
        budget = 20 * epsilon_inf # select a random total privacy budget for each client
        c = Client(epsilon_inf, epsilon_1, m, budget, thresh, part)  # create client
        for _ in range(10):  # number of the client's reports  
            x = np.random.randint(0, m)  # select randomly the counter value of the client
            input_data.append(x)
            real[int(x * k / m)] += 1  # calculate real frequency
            report = c.report(x)  # passing the data to the client and getting its report
            s.collect(report)  # passing the client's report to the server
        responses.append(c.rep)

    for i in range(m):  # calculate histogram estimation error
        avg += i * s.estimation(i)
        estimate[int(i * k / m)] += s.estimation(i)

    real /= np.sum(responses)

    con_b.append(np.mean(consumed_budget))
    avg_err.append(abs(np.mean(input_data) - avg))
    MSE.append(mean_squared_error(estimate, real))
    MAE.append(mean_absolute_error(estimate, real))

print("with ep=%f:  MAE: %f  MSE: %f  AVG CB: %f"
      % (ep+1, np.mean(MAE), np.mean(MSE), np.mean(con_b)))

plot(real, estimate, int(m/k))
