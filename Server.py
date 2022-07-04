import numpy as np


class Server:

    def __init__(self, epsilon_inf, epsilon_1, m):
        self.epsilon_inf = epsilon_inf
        self.epsilon_1 = epsilon_1
        self.m = m  # client's counter values are integers in [0, m)
        self.reports = []  # client's reports

    # collect client's reports
    def collect(self, rep):
        self.reports.append(rep)

    # calculate histogram estimation
    def estimation(self, v):
        h = 0  # number of reports with value v
        n = len(self.reports)  # number of reports
        e_inf = np.exp(self.epsilon_inf)
        e_1 = np.exp(self.epsilon_1)
        p1 = e_inf / (e_inf + self.m - 1)
        q1 = (1 - p1) / (self.m - 1)
        p2 = (e_1*e_inf-1) / ((self.m-1)*e_inf - self.m*e_1 + e_1 + e_1*e_inf - 1)
        q2 = (1 - p2) / (self.m - 1)
        for rep in self.reports:
            if rep == v:  h += 1
        estimate = ((h - n*q1*(p2-q2) - n*q2) / (n * (p1-q1)*(p2-q2))).clip(0)
        return estimate
