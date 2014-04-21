import numpy as np
import theano
import theano.tensor as T
import dct
import scipy.stats
import PIL.Image
from utils import tile_raster_images

rng = np.random

def sparsityCost(layer, spar=.1):
    # Col-Mean: AvgAct of each hidden unit across all m-examples
    avgAct = layer.getActivation().mean(axis=0) 
    return T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))

def reconstructionCost(layer, x):
    a = layer.getActivation()
    return T.sum((a - x) ** 2) / (2. * x.shape[0])

def weightCost(layer):
    W = layer.getWeights()
    return T.sum(W**2)    

def xentCost(layer, y):
    p_y_given_x = layer.getActivation()
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    return cost

class Layer(object):
    def __init__(self, inputSize, outputSize, activation=T.nnet.sigmoid):
        self.inputSize     = inputSize
        self.outputSize    = outputSize
        self.actFn         = activation
        self.nWeightParams = inputSize * outputSize
        self.nBiasParams   = outputSize
        self.nParams       = self.nWeightParams + self.nBiasParams

    def getInputSize  (self): return self.inputSize
    def getOutputSize (self): return self.outputSize
    def getNParams    (self): return self.nParams
    def getActivation (self): return self.act
    def getWeights    (self): return self.W

    def getx0(self):
        r = np.sqrt(6) / np.sqrt(self.inputSize+self.outputSize+1)
        return np.concatenate(((rng.randn(self.nWeightParams)*2*r-r).flatten(),
                               np.zeros(self.nBiasParams)))
        
    def setTheta(self, theta):
        self.W = theta[:self.inputSize*self.outputSize].reshape((self.inputSize, self.outputSize))
        self.b = theta[self.inputSize*self.outputSize:]

    def getTheta(self):
        if type(self.W) == np.ndarray:
            return np.concatenate([self.W.flatten(),self.b])
        else:
            return np.concatenate([self.W.eval().flatten(),self.b.eval()])

    def forward(self, x):
        if not self.actFn:
            self.act = T.dot(x, self.W) + self.b
        else:
            self.act = self.actFn(T.dot(x, self.W) + self.b)
        return self.act

    def saveImage(self, fname):
        z = self.W.T if type(self.W) == np.ndarray else self.W.eval().T
        image = PIL.Image.fromarray(tile_raster_images(
                X=z,
                img_shape=(28, 28), tile_shape=(14, 14),
                tile_spacing=(1, 1)))
        image.save(fname)

    def __str__(self):
        shape = self.W.shape if type(self.W) == np.ndarray else self.W.eval().shape
        return "%s Activation %s; %d parameters"%(shape,self.actFn,self.nParams)
    
class Softmax(Layer):
    def __init__(self, inputSize, nClasses):
        super(Softmax,self).__init__(inputSize, nClasses, T.nnet.softmax)

    def accuracy(self, y):
        p_y_given_x = self.getActivation()
        y_pred = T.argmax(p_y_given_x, axis=1)
        accuracy = T.mean(T.neq(y_pred, y))
        return accuracy

