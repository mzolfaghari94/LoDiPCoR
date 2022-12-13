import numpy as np


class Server:

    def __init__(self, epsilon_inf, epsilon_1):
        if epsilon_1 >= epsilon_inf:
            raise ValueError('Please set epsilon_1 < epsilon_infinity')
        else:
            self.f = 2 / (1 + np.exp(epsilon_inf / 2))
            self.p = (np.exp(epsilon_inf / 2) - np.exp(epsilon_1 / 2)) / \
                     ((np.exp(epsilon_inf / 2) - 1) * (np.exp(epsilon_1 / 2) + 1))
            self.q = 1 - self.p

            if (np.array([self.f, self.p, self.q]) >= 0).all():
                pass
            else:
                raise ValueError('Probabilities are negative.')

    def estimate(self, reports):
        if len(reports) == 0:
            raise ValueError('List of reports is empty.')
        n = len(reports)
        est_freq = ((sum(reports) - n * (self.p + (self.f / 2) * (self.q - self.p))) /
                    ((1 - self.f) * (self.q - self.p))).clip(0)
        if sum(est_freq) > 0:
            est_freq = np.nan_to_num(est_freq / sum(est_freq))
        return est_freq
