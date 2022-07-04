import numpy as np
from Client import Client
from Server import Server
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# epsilon_1 = 1  # privacy budget in each instantaneous randomized response
# epsilon_inf = 2  # privacy budget in each permanent randomized response
clients: int = 10000  # number of clients
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
    for ep in range(10):
        epsilon_1 = ep+1
        epsilon_inf = epsilon_1 * 2
        part = 0.6
        MAE = []
        MSE = []
        avg_err = []
        con_b = []
        thresh = 0.5

        for rnd in range(50):
            error = []  # estimating error for each value
            initial_budget = []  # initial budget of each client
            input_data = []
            end_budget = []  # end budget of each client
            consumed_budget = []  # consumed budget of each client
            responses = []  # number of responses of each client
            bc_responses = []  # number of budget-consuming responses of each client
            nbc_responses = []  # number of non-budget-consuming responses of each client
            real = np.zeros(k)  # real histogram
            estimate = np.zeros(k)  # real histogram
            s = Server(epsilon_inf * part, epsilon_1, m)  # create the server
            avg = 0
            # np.random.randint(1, 2)
            for _ in range(clients):
                # select a random total privacy budget for each client
                budget = 20 * epsilon_inf
                c = Client(epsilon_inf, epsilon_1, m, budget, thresh, part)  # create client
                for x in range(10):  # select number of the client's reports randomly from 1 to 9
                    # x = np.random.randint(0, m)  # select randomly the counter value of the client
                    input_data.append(x)
                    real[int(x * k / m)] += 1  # calculate real frequency
                    report = c.report(x)  # passing the data to the client and getting its report
                    s.collect(report)  # passing the client's report to the server
                initial_budget.append(budget)
                end_budget.append(c.total_budget)
                consumed_budget.append(budget - c.total_budget)
                responses.append(c.rep)
                bc_responses.append(c.bc)
                nbc_responses.append(c.nbc)

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

    # plot(real, estimate, int(m/k))
