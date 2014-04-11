import numpy as np
import theano
import theano.tensor as T
import dct
import PIL.Image
from utils import tile_raster_images

rng = np.random

def xentCost(p_y_given_x, y):
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    return cost

class Softmax():
    def __init__(self, visibleSize, nClasses):
        self.visibleSize = visibleSize
        self.nClasses = nClasses
        self.nWeightParams = visibleSize * nClasses
        self.nBiasParams = nClasses
        self.nParams = self.nWeightParams + self.nBiasParams

    def getNParams(self):
        return self.nParams

    def getx0(self):
        r = np.sqrt(6) / np.sqrt(self.nWeightParams)
        return np.concatenate(((rng.randn(self.nWeightParams)*2*r-r).flatten(),
                               np.zeros(self.nBiasParams)))

    def setTheta(self, theta):
        self.W = theta[:self.visibleSize*self.nClasses].reshape((self.visibleSize,self.nClasses))
        self.b = theta[self.visibleSize*self.nClasses:]

    def forward(self, x):
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return p_y_given_x

    def accuracy(self, p_y_given_x, y):
        y_pred = T.argmax(p_y_given_x, axis=1)
        accuracy = T.mean(T.neq(y_pred, y))
        return accuracy

    def cost(self, x, output, labels):
        return xentCost(output, labels)

    def __str__(self):
        return "Softmax Classifier. %d parameters"%self.nParams
