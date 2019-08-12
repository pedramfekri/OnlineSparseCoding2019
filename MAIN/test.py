import ODL
import numpy as np
from MAIN.SparseCoding import SparseCodingClass


# S = np.random.rand(200,1)
# D = np.random.rand(400,200)
# B = np.random.rand(400,200)
# A = np.random.rand(200,200)
# X = np.random.rand(400,1)
# x = ODL.ODLClass(S, D, A, B, X)
# x.DictionaryUpdate()

# print(x.D)

x = SparseCodingClass(50, 0.9, 100)
x.initialize('numeric')
x.execute()
d = x.getDictionary()
s = x.getFeature()
print(x.getDictionary().shape)
# x.exec()
# print(x.Y[0])
# im = np.random.rand(4,4)
# r = x.imPatcher(im)
# print(im,'\n')
# print(r[0],"\n")
# print(r[1],'\n')
# print(r[2],'\n')
# print(r[3],'\n')
# print(r[0].reshape([2,2]),"\n")
# print(r[1].reshape([2,2]),'\n')
# print(r[2].reshape([2,2]),'\n')
# print(r[3].reshape([2,2]),'\n')







