import numpy as np


class FeatureLearningClass:
    t = 1
    stepsize = 1
    itr = 0
    dif = 10
    obn = 10
    plx = 0
    C = 0

    def __init__(self):
        self.C = 0
        self.x = 0
        self.im = 0
        self.lamb = 0
        self.plx = 0
        self.y = 0
        self.itr = 0

    def initialize(self, C, x, im, lamb, itr):
        self.C = C
        self.x = x
        self.im = im
        self.lamb = lamb
        self.plx = x
        self.y = x
        self.itr = itr

    def FISTAOptimize(self):

        while (self.dif > 10 ** -3) & (self.itr <= 1500):
            plx0 = self.plx
            r = self.backTrackingFISTA(self.y, self.stepsize)
            self.stepsize = r[1]
            self.plx = r[0]
            tp = self.t
            self.t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
            self.y = self.plx + ((tp - 1) / (self.t)) * (self.plx - plx0)
            self.itr = self.itr + 1
            obn0 = self.obn
            self.obn = self.objectiveValue(self.y)
            self.dif = np.abs(obn0 - self.obn)
            # print(self.obn, self.itr ,'\n')

        self.x = self.plx
        return self.plx

    def objectiveValue(self, x):
        r = np.linalg.norm(self.C.dot(x) - self.im,2)**2 + self.lamb * np.linalg.norm(x,1)
        return r

    def backTrackingFISTA(self, prex, prestsize):
        gamma = 0.1
        newstsize = prestsize
        gradval = self.gradient(prex)
        newx = self.shrinkage(self.lamb*newstsize, prex - newstsize * gradval)
        counter = 0
        diff = self.f(newx) - self.f(prex) - (newx - prex).transpose().dot(gradval) - 1 / (2*newstsize)* np.linalg.norm(newx - prex,2)**2

        while diff > 0 & counter < 100:
            counter = counter + 1
            newstsize = newstsize * gamma
            newx = self.shrinkage(self.lamb*newstsize, prex - newstsize*gradval)
            diff = self.f(newx) - self.f(prex) - (newx - prex).transpose().dot(gradval) - 1 / (
            2 * newstsize) * np.linalg.norm(newx - prex, 2) ** 2

        return (newx, newstsize)

    def shrinkage(self, thresh, vec):
        for i in range(vec.size):
            if vec[i] > thresh:
                vec[i] = vec[i] - thresh
            elif vec[i] < -thresh:
                vec[i] = vec[i] + thresh
            else:
                vec[i] = 0

        return vec

    def gradient(self,x):
        r = self.C.transpose().dot((self.C.dot(x) - self.im))
        return r

    def f(self, x):
        r = 0.5 * np.linalg.norm(self.C.dot(x) - self.im, 2)**2
        return r

    def g(self, x):
        r = self.lamb * np.linalg.norm(x,1)
        return r

    def Q(self, nx, px ,L):
        def_nx_px = nx - px
        def_nx_px_tran = def_nx_px.transpose()
        r = self.f(px)\
            + def_nx_px_tran.dot(self.gradient(px))\
            + (L/2)\
            * np.linalg.norm(def_nx_px,2)**2\
            + self.g(nx)
        return r

    def setDictionary(self, D):
        self.C = D

    def setFeature(self, x):
        self.x = x

    def setInput(self, y):
        self.im = y

    def getFeature(self):
        return self.x



