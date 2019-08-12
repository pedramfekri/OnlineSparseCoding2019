from SparseCoding import SparseCodingClass
import matplotlib.image as m
import matplotlib.pyplot as im
import numpy as np
a = SparseCodingClass(500,0.1,10)
a.initialize("image")
a.execute()
sum(a.Y[1].x == 0)
sum(a.Y[1].x[:, 1] == 0)
a.Y[1].x[:, 1]
d = np.matmul(a.D, a.Y[1].x[:, 1])
im.imshow(a.col2im(d, 20, 20))
im.imshow(a.col2im(a.Y[1].inp[:,1], 20, 20))