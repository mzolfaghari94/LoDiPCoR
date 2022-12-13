import numpy as np


class Client:

    def __init__(self, epsilon):
        self.a_m_t = None
        self.R = []
        self.previousValue = 0
        self.t = 0
        self.m_t_1 = 0
        self.epsilon = epsilon
        self.changes = 0

    def newValue(self, v):
        self.t = self.t + 1
        aArray = self.leafNodesPerTree(self.t)
        aArray = aArray.astype(np.int64)
        previousR = self.R.copy()
        a_1 = aArray[0]
        self.a_m_t = aArray[len(aArray) - 1]
        self.R = [previousR[j]
                  if j > self.a_m_t
                  else np.sum(previousR[0:j]) + v - self.previousValue
                  for j in range(a_1 + 1)]
        if self.previousValue != v:
            self.changes += 1
        self.previousValue = v

    def leafNodesPerTree(self, numberOfNodes):
        if numberOfNodes == 0:
            return []
        leafNodesInCurrentTree = np.floor(np.log2(numberOfNodes))
        remaningNodes = numberOfNodes - np.pow(2, leafNodesInCurrentTree)
        return np.concatenate(([leafNodesInCurrentTree], self.leafNodesPerTree(remaningNodes)))

    def select(self):
        r_t = self.a_m_t
        h_t = np.random.randint(0, 2)
        if h_t >= 1:
            h_t = r_t
        else:
            h_t = 0
        return [self.R[h_t], h_t]

    def perturbation(self, v):
        rand = np.random.random()
        if v == 0:
            if rand < 0.5:
                return 1
            else:
                return -1
        else:
            setToOneP = 0.5 + (v / 2) * ((np.exp(self.epsilon) - 1) / (np.exp(self.epsilon) + 1))
            if rand < setToOneP:
                return 1
            else:
                return -1

    def report(self, data):
        self.newValue(data)
        [v, h] = self.select()
        v = self.perturbation(v)
        return [v, h]


class WrappedClient:
    def __init__(self, m, epsilon):
        self.m = m
        self.epsilon = epsilon
        self.Clients = [Client(epsilon) for _ in range(m)]
        self.changes = 0
        self.prevValue = -1

    def report(self, value):
        if value != self.prevValue:
            self.changes += 1
        self.prevValue = value
        binaryRepresentation = f'{value:0{self.m}b}'
        characterized = [c for c in binaryRepresentation]
        allV = []
        allH = []
        for i in range(self.m):
            toReport = int(characterized[i])
            [v, h] = self.Clients[i].report(toReport)
            allV.append(v)
            allH.append(h)
        return [allV, allH]

    def consumedBudget(self):
        return self.changes * self.epsilon
