import numpy


class Client:

    def __init__(self, epsilon, m, s, gamma):
        self.epsilon = epsilon
        self.m = m
        self.gamma = gamma
        self.bulk_size = s
        self.alpha = numpy.random.randint(0, s)
        self.cache = dict()
        self.budget = 0

    # calculate and report the privatized value
    def report(self, x):
        y = self.round(x)
        # if value is not memoized, calculate the PRR
        if self.cache.get(y) is None:

            bit = self.oneBit(y)
            self.cache[y] = bit
            self.budget += self.epsilon
        else:
            bit = self.cache.get(y)

        """
        IRR: flip the output of memoized responses with a small probability 0 ≤ gamma ≤ 0.5
        which ensures that data collector will not be able to learn with certainty that
        behavior of a user changed at certain time stamps.
        """
        if numpy.random.random() <= self.gamma:
            bit = bit - 1
        return bit

    # calculate 1-Bit Mechanism for mean estimation
    def oneBit(self, y):
        prob = (1 / (numpy.exp(self.epsilon) + 1)) + (y / self.m) * (
                    (numpy.exp(self.epsilon) - 1) / (numpy.exp(self.epsilon) + 1))
        bit = 0
        if numpy.random.random() <= prob:  bit = 1
        return bit

    # calculate α-point rounding
    def round(self, x):
        bulk = int(x / self.bulk_size)
        l = bulk * self.bulk_size
        r = (bulk + 1) * self.bulk_size
        if x + self.alpha < r: return l
        return r
