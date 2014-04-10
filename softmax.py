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
        nWeightParams = visibleSize * nClasses
        nBiasParams = nClasses
        nParams = nWeightParams + nBiasParams
        print "Softmax Classifier\n%d total parameters"%nParams
        self.theta = theano.shared(value=np.zeros(nParams,dtype=theano.config.floatX),name='theta',borrow=True)
        self.W = self.theta[:visibleSize*nClasses].reshape((visibleSize,nClasses))
        self.b = self.theta[visibleSize*nClasses:]

        r = np.sqrt(6) / np.sqrt(nWeightParams)
        self.x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams))).astype('float32')

    def forward(self, x):
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        return p_y_given_x

    def getAccuracy(self, p_y_given_x, y):
        y_pred = T.argmax(p_y_given_x, axis=1)
        accuracy = T.mean(T.neq(y_pred, y))
        return accuracy
