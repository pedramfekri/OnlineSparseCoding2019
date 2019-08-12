from OnlineDictionaryLearning.ODL import ODLClass
from FeatureLearning.FL import FeatureLearningClass
from MAIN.InputClass import InputFrame
import numpy as np
import matplotlib.image as m
import matplotlib.pyplot as im
import os
import matplotlib.pyplot as plt
from skimage import color
from skimage.transform import resize


class SparseCodingClass:
    path = ""
    Y = []
    D = 0
    B = 0
    A = 0
    x = 0
    itr = 100
    sparsity = 0
    # it must be 4 for images inputs
    patch = 4
    overLap = 0

    def __init__(self, dim, sparsity, itr):
        self.dim = dim
        self.sparsity = sparsity
        self.itr = itr

    def initialize(self, type):
        # s = float(input("please enter your name:"))
        # s = s + 1.1
        # print("your name is ", s)
        if type == 'image':
            self.path = """C:\\Thesis\\Online Sparse Coding - Variable Dim\\10Classes\\"""
            i = 0
            im = []
            for filename in os.listdir(self.path):
                # print(os.path.abspath(path+filename),'\n')
                # im.append(color.rgb2grey(m.imread(os.path.abspath(self.path+filename))))
                # print(im[i])
                # i = + 1
                gray_im = color.rgb2grey(m.imread(os.path.abspath(self.path+filename)))
                gray_im_resized = resize(gray_im, [40, 40])
                patches = self.imPatcher(gray_im_resized)
                input_object = InputFrame(self.patch, self.dim, len(patches[0]))
                input_object.inputFiller(patches)
                im.append(input_object)

            self.D = np.random.rand(len(patches[0]), self.dim)
            self.B = np.zeros([len(patches[0]), self.dim])
            self.A = np.zeros([self.dim, self.dim])
            self.Y = im
        else:

            file = np.loadtxt("""C:\\Users\\Pedram\\PycharmProjects\\onlineSparseCodingAugmentation\\MAIN\\P_Brandt_1_Overheated.csv""", delimiter=",")
            # return (file.shape, file)
            # self.path = """C:\\Thesis\\Online Sparse Coding - Variable Dim\\10Classes\\"""
            im = []
            for i in range(file.shape[0]):
                data = file[i, :]
                input_object = InputFrame(self.patch, self.dim, len(data))
                input_object.inputFiller(data)
                im.append(input_object)

            self.D = np.random.rand(len(data), self.dim)
            self.B = np.zeros([len(data), self.dim])
            self.A = np.zeros([self.dim, self.dim])
            self.Y = im


        # a = plt.imshow(im[0])
        # # plt.show(a)
        # print(im[0].shape)

    def execute(self):


        # iteration in range of input number
        for f in range(20):
            # iteration in rang of input patches number
            for p in range(self.Y[f].inp.shape[1]):
                # iteration in rang of input number
                for i in range(self.itr):
                    y = self.Y[f].inp[:, p]
                    x = self.Y[f].x[:, p]
                    featureLearn = FeatureLearningClass()
                    featureLearn.initialize(self.D, x, y, self.sparsity, 50)
                    featureLearn.FISTAOptimize()
                    x = featureLearn.getFeature()
                    self.Y[f].x[:, p] = x
                    dictionaryLearn = ODLClass()
                    dictionaryLearn.initialize(x, self.D, self.A, self.B, y)
                    dictionaryLearn.DictionaryUpdate()
                    self.A = dictionaryLearn.get_a()
                    self.D = dictionaryLearn.get_d()
                    self.B = dictionaryLearn.get_b()
                    # print('input: ', f, 'reconstruction error = ', self.reconstructionError(x, y, self.D))
            print('input: ', f)

        return (self.Y, self.D)

    def setSparsity(self, lamda):
        self.sparsity = lamda

    def getSparsity(self):
        return self.sparsity

    def setDimension(self,dim):
        self.dim = dim

    def getDimension(self):
        return self.dim

    def getFeature(self):
        return self.Y

    def getDictionary(self):
        return self.D

    def setPatchSize(self, size):
        self.patch = size

    def getPatchSize(self):
        return self.patch

    def setOverlapSize(self, size):
        self.overLap = size

    def getOverlapSize(self):
        return self.overLap

    def imPatcher(self, im):
        p1 = im[0:int((im.shape[0]/2)), 0:int((im.shape[1]/2))]
        p2 = im[0:int((im.shape[0]/2)), int((im.shape[1]/2)):im.shape[1]]
        p3 = im[int((im.shape[0]/2)):im.shape[0], 0:int((im.shape[1]/2))]
        p4 = im[int((im.shape[0]/2)):im.shape[0], int((im.shape[1]/2)):im.shape[1]]
        return (self.im2col(p1), self.im2col(p2), self.im2col(p3), self.im2col(p4))

    def im2col(self, a):
        b = a.reshape(a.shape[0] * a.shape[1])
        return b

    def col2im(self, vec, row ,col):
        b = vec.reshape([row, col])
        return b

    def reconstructionError(self, x, y, D):
        r = np.linalg.norm(y - D.dot(x), 2)**2
        return r

    def getInputFeature(self):
        return self.Y




