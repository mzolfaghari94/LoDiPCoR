import numpy as np


class Client:

    def __init__(self, epsilon_inf, epsilon_1, m):
        if epsilon_1 >= epsilon_inf:
            raise ValueError('Please set epsilon_1 < epsilon_infinity')
        else:
            self.f = 2 / (1 + np.exp(epsilon_inf / 2))
            self.p = (np.exp(epsilon_inf / 2) - np.exp(epsilon_1 / 2)) / \
                     ((np.exp(epsilon_inf / 2) - 1) * (np.exp(epsilon_1 / 2) + 1))
            self.q = 1 - self.p
            self.m = m
            self.budget = 0
            self.epsilon = epsilon_inf

            if (np.array([self.f, self.p, self.q]) >= 0).all():
                pass
            else:
                raise ValueError('Probabilities are negative.')

    def ue(self, input_data):
        ue_data = np.zeros((len(input_data), self.m))
        i = 0
        for data in input_data:
            ue_data[i][data] = 1
            i += 1
        return ue_data

    def round1(self, ue_data, k):
        first_sanitization = np.zeros((k, self.m))
        for i in range(len(ue_data)):
            for j in range(i):
                if (ue_data[i] == ue_data[j]).all():
                    first_sanitization[i] = first_sanitization[j]
                    break
            if (first_sanitization[i] == np.zeros(self.m)).all():
                self.budget += self.epsilon
                for j in range(self.m):
                    rnd = np.random.random()
                    if rnd < self.f:
                        first_sanitization[i][j] = np.random.randint(0, 2)
                    else:
                        first_sanitization[i][j] = ue_data[i][j]
        return first_sanitization

    def round2(self, s_data, k):
        second_sanitization = np.zeros((k, self.m))
        i = 0
        for first_data in s_data:
            for ind in range(self.m):
                rnd = np.random.random()
                if first_data[ind] == 1:
                    if rnd <= self.q:
                        second_sanitization[i][ind] = 1
                else:
                    if rnd <= self.p:
                        second_sanitization[i][ind] = 1
            i += 1
        return second_sanitization

    def report(self, input_data):
        ue_data = self.ue(input_data)
        first_sanitization = self.round1(ue_data, len(input_data))
        second_sanitization = self.round2(first_sanitization, len(input_data))
        return second_sanitization
