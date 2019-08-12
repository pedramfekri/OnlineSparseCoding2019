import numpy as np


class ODLClass:
    def __init__(self):
        self.S = 0
        self.D = 0
        self.A = 0
        self.B = 0
        self.X = 0

    def initialize(self, S, D, A, B, X):
        self.S = S.reshape(len(S),1)
        self.D = D
        self.A = A
        self.B = B
        self.X = X.reshape(len(X),1)

    def D_Row(self):
        return self.D.shape[0]

    def D_Col(self):
        return self.D.shape[1]

    def S_Row(self):
        return self.S.shape[0]

    def S_Col(self):
        return self.S.shape[1]

    def A_Row(self):
        return self.A.shape[0]

    def A_Col(self):
        return self.A.shape[1]

    def B_Row(self):
        return self.B.shape[0]

    def B_Col(self):
        return self.B.shape[1]

    def X_Row(self):
        return self.X.shape[0]

    def X_Col(self):
        return self.X.shape[1]

    #set and get parameter

    def set_d(self, inputs):
        self.D = inputs

    def get_d(self):
        return self.D

    def set_s(self, inputs):
        self.S = inputs

    def get_s(self):
        return self.S

    def set_a(self, inputs):
        self.A = inputs

    def get_a(self):
        return self.A

    def set_b(self, inputs):
        self.B = inputs

    def get_b(self):
        return self.B

    def set_x(self, inputs):
        self.X = inputs

    def get_x(self):
        return self.X

    def DictionaryUpdate(self):

        self.A = self.A + self.S.dot(self.S.transpose())
        self.B = self.B + self.X.dot(self.S.transpose())

        for i in range(self.D_Col()):
            u = (1/self.A[i, i]) * (self.B[:, i] - self.D.dot(self.A[:, i])) + self.D[:, i]
            # pay attention to this line please
            u = u.reshape(len(self.X),1)

            if ~np.isnan(u[1]):
                self.D[:, i] = (1 / np.max([np.linalg.norm(u), 1])) * u[:, 0]

        return (self.D , self.B, self.A)


