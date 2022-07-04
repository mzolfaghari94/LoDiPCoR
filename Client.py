import numpy as np


class Client:

    def __init__(self, epsilon_inf, epsilon_1, m, budget, thresh, part):
        self.epsilon_inf = epsilon_inf * part
        self.epsilon_rr = epsilon_inf - self.epsilon_inf
        self.epsilon_1 = epsilon_1
        self.m = m  # client's counter values are integers in [0, m)
        self.total_budget = budget  # total privacy budget for every client
        self.thresh = thresh
        self.cache = dict()  # cache for memoization
        self.rep = 0  # number of responses of each client
        self.bc = 0  # number of budget-consuming responses of each client
        self.nbc = 0  # number of non-budget-consuming responses of each client

    # calculate and report the privatized value
    def report(self, x):
        self.rep += 1
        if self.cache.get(x) is None:  # if value is not memoized, calculate the PRR
            y = self.cache[x] = self.prr(x)  # calculate the PRR and memoize it
        else:
            y = self.cache.get(x)  # get memoized value
        e_inf = np.exp(self.epsilon_inf)
        e_1 = np.exp(self.epsilon_1)
        prob = (e_1*e_inf - 1) / ((self.m-1)*e_inf - self.m*e_1 + e_1 + e_1*e_inf - 1)
        return self.grr(y, prob)  # calculate the IRR and return

    # find the closest last released value to x
    def closest(self, x):
        min_distance = self.m  # minimum distance of the last released values to x
        closest_to_x = None  # for the first time, the closest value is x-1
        for item in self.cache:
            distance = abs(item - x)
            if distance < min_distance:
                min_distance = distance
                closest_to_x = item
        return closest_to_x

    # is_close
    def is_close(self, x, y):
        if y is None: return 0
        real = 1 if abs(x - y) < self.thresh else 0
        prob = np.exp(self.epsilon_rr) / (np.exp(self.epsilon_rr) + 1)
        if np.random.random() < prob:
            return real
        return 1 - real

    # calculate Permanent Randomized Response
    def prr(self, x):
        if self.total_budget - self.epsilon_inf >= 0:
            self.total_budget -= self.epsilon_rr  # calculate new total budget of this client
            y = self.closest(x)
            if self.is_close(x, y) == 1:
                return self.cache.get(y)
            self.total_budget -= self.epsilon_inf  # calculate new total budget of this client
            self.bc += 1
            prob = np.exp(self.epsilon_inf) / (np.exp(self.epsilon_inf) + self.m - 1)
            return self.grr(x, prob)

        # if the new total budget is smaller than zero, use memoized PRR for its closest value
        return self.cache.get(self.closest(x))

    # calculate Generalized Randomized Response
    def grr(self, x, p):
        if np.random.random() < p: return x
        return np.random.choice(list(range(0, x)) + list(range(x + 1, self.m)))
