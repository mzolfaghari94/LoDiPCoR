import numpy as np


class Server:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.coef = (1 + np.exp(self.epsilon)) / (np.exp(self.epsilon) - 1)
        self.sumVOf1 = 0
        self.sumOfUsersOf1 = 0
        self.sumVOfh = 0
        self.sumOfUsersOfh = 0
        self.f1 = 0
        self.f2 = 0
        self.f = [0.0]
        self.varianceF = [0.0]
        self.t = 0
        self.lastRoot = 0

    def newValue(self, v, h):
        callibratedV = v * self.coef
        self.lastRoot = max(self.lastRoot, h)
        if h == 0:
            self.sumOfUsersOf1 += 1
            self.sumVOf1 += callibratedV
        else:
            self.sumOfUsersOfh += 1
            self.sumVOfh += callibratedV

    def varianceF1(self):
        varF1 = self.varianceF[len(self.varianceF) - 1] + \
                (((np.exp(self.epsilon) + 1) / (np.exp(self.epsilon) - 1)) ** 2) / \
                self.sumOfUsersOf1
        return varF1

    def varianceF2(self):
        tPrime = self.t - 2 ** self.lastRoot
        varF2 = self.varianceF[tPrime] + (((np.exp(self.epsilon) + 1) / (np.exp(self.epsilon) - 1)) ** 2) / \
                self.sumOfUsersOfh
        return varF2

    def computeVariance(self):
        if self.t % 2 == 0:
            vf1 = self.varianceF1()
            vf2 = self.varianceF2()
            return (vf1 * vf2) / (vf1 + vf2)
        else:
            return self.varianceF[len(self.varianceF) - 1] + \
                   (((np.exp(self.epsilon) + 1) / (np.exp(self.epsilon) - 1)) ** 2) / self.sumOfUsersOf1

    def computeW1(self):
        varf1 = self.varianceF1()
        return np.pow(varf1, -1)

    def computeW2(self):
        varf2 = self.varianceF2()
        return np.pow(varf2, -1)

    def computeW(self):
        w1 = self.computeW1()
        w2 = self.computeW2()
        return w1 / (w1 + w2)

    def predicate(self):
        self.t += 1
        if self.t % 2 == 0:
            self.f1 = self.f[len(self.f) - 1] + (self.sumVOf1 / self.sumOfUsersOf1)
            tPrime = self.t - 2 ** self.lastRoot
            self.f2 = self.f[tPrime] + (self.sumVOfh / self.sumOfUsersOfh)
        else:
            self.f1 = self.f2 = self.f[len(self.f) - 1] + (self.sumVOf1 / self.sumOfUsersOf1)
        if self.t % 2 == 0:
            w = self.computeW()
        else:
            w = 0.5  # Just to neutralize its effect.
        freq = w * self.f1 + (1 - w) * self.f2
        self.f.append(freq)
        varF = self.computeVariance()
        self.varianceF.append(varF)

        # Reset state of server.
        self.sumVOf1 = 0
        self.sumOfUsersOf1 = 0
        self.sumVOfh = 0
        self.sumOfUsersOfh = 0

    def finish(self):
        self.f = np.clip(self.f, 0, 1)
        return self.f


class WrappedServer:
    def __init__(self, M, epsilon):
        self.M = M
        self.epsilon = epsilon
        self.servers = [Server(epsilon) for _ in range(M)]

    def newValue(self, v, h, m):
        self.servers[m].newValue(v, h)

    def predicate(self):
        for server in self.servers:
            server.predicate()

    def finish(self):
        toDetermineShape = self.servers[0].finish()
        result = np.zeros([len(toDetermineShape), self.M])
        for index, server in enumerate(self.servers):
            result[:, index] = server.finish()
        return result[1:]
