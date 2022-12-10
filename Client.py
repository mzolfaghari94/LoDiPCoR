import numpy as np


class Client:

    def __init__(self, epsilon_inf, epsilon_1, m, th, b):
        if epsilon_1 >= epsilon_inf:
            raise ValueError('Please set epsilon_1 < epsilon_infinity')
        else:
            self.f = 2 / (1 + np.exp(epsilon_inf / 2))
            self.p = (np.exp(epsilon_inf / 2) - np.exp(epsilon_1 / 2)) / \
                     ((np.exp(epsilon_inf / 2) - 1)*(np.exp(epsilon_1 / 2) + 1))
            self.m = m
            self.epsilon = epsilon_inf
            self.thresh = th
            self.budget = b

            if (np.array([self.f, self.p, 1 - self.f, 1 - self.p]) >= 0).all():
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

    def closest(self, i, data):
        min_distance = self.m
        closest_to_i = i
        for j in range(i):
            distance = abs(data[i] - data[j])
            if distance < min_distance:
                min_distance = distance
                closest_to_i = j
        return closest_to_i, min_distance

    def is_memoized(self, x, data):
        for j in range(x):
            if (data[x] == data[j]).all():
                return j
        return -1

    def round1(self, ue_data, input_data):
        first_sanitization = np.zeros((len(input_data), self.m))
        for i in range(len(ue_data)):
            memoized = self.is_memoized(i, ue_data)
            close, distance = self.closest(i, input_data)
            if memoized >= 0:
                first_sanitization[i] = first_sanitization[memoized]
            elif (self.budget - self.epsilon) < 0 or distance <= self.thresh:
                first_sanitization[i] = first_sanitization[close]
            else:
                self.budget -= self.epsilon
                for j in range(self.m):
                    rnd = np.random.random()
                    if rnd < self.f:
                        first_sanitization[i][j] = np.random.randint(0, 2)
                    else:
                        first_sanitization[i][j] = ue_data[i][j]
        return first_sanitization

    def round2(self, s_data, k):
        second_sanitization = np.zeros((k, self.m))
        for i in range(len(s_data)):
            for j in range(self.m):
                rnd = np.random.random()
                if s_data[i][j] == 0 and rnd < self.p:
                    second_sanitization[i][j] = 1
                elif s_data[i][j] == 1 and rnd < (1 - self.p):
                    second_sanitization[i][j] = 1
        return second_sanitization

    def report(self, input_data):
        ue_data = self.ue(input_data)
        first_sanitization = self.round1(ue_data, input_data)
        second_sanitization = self.round2(first_sanitization, len(input_data))
        return second_sanitization
