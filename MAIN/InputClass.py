import numpy as np


class InputFrame:

    x = []
    inp = []

    def __init__(self, patch, featureSize, inputSize):
        self.x = np.random.rand(featureSize, patch)
        self.inp = np.zeros([inputSize, patch])

    def inputFiller(self, patches):
        if self.x.shape[1] == 1:
            self.inp[:, 0] = patches
        else:
            for i in range(self.x.shape[1]):
                self.inp[:, i] = patches[i]
