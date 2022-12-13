import numpy as np


class Server:

    def __init__(self, epsilon, m, gamma):
        self.epsilon = np.log(((1 - 2 * gamma)*(np.exp(epsilon) / (np.exp(epsilon) + 1)) + gamma) /
                              ((1 - 2 * gamma)*(1 / (np.exp(epsilon) + 1)) + gamma))
        self.epsilon = epsilon
        self.m = m
        self.bits = []

    # collect client's reports into array bits
    def collect(self, bit):
        self.bits.append(bit)

    # calculate mean estimation
    def estimation(self):
        mean = 0
        for bit in self.bits:
            mean += bit * (np.exp(self.epsilon) + 1) - 1
        return mean * self.m / (len(self.bits) * (np.exp(self.epsilon) - 1))
