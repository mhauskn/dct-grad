import numpy as np
import theano
import theano.tensor as T
import dct
import scipy.stats
import PIL.Image
from utils import tile_raster_images

rng = np.random

### Cost Functions

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
    cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, y))
    #cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    return cost

class Layer(object):
    def __init__(self, inputShape, outputShape, activation=T.nnet.sigmoid):
        '''
            inputShape may be a tuple of form [nBatches, inputSize] if passed
            from a convolutional layer.
        '''
        if type(inputShape) is tuple:
            assert len(inputShape) < 3
            inputShape = inputShape[-1]
        self.inputShape    = inputShape
        self.outputShape   = outputShape
        self.actFn         = activation
        self.nWeightParams = inputShape * outputShape
        self.nBiasParams   = outputShape
        self.nParams       = self.nWeightParams + self.nBiasParams

    def getInputShape (self): return self.inputShape
    def getOutputShape(self): return self.outputShape
    def getNParams    (self): return self.nParams
    def getActivation (self): return self.act
    def getWeights    (self): return self.W

    def getx0(self):
        r = np.sqrt(6) / np.sqrt(self.inputShape+self.outputShape+1)
        return np.concatenate(((rng.randn(self.nWeightParams)*2*r-r).flatten(),
                               np.zeros(self.nBiasParams)))
        
    def setTheta(self, theta):
        self.W = theta[:self.inputShape*self.outputShape].reshape((self.inputShape, self.outputShape))
        self.b = theta[self.inputShape*self.outputShape:]

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
        a = int(np.sqrt(self.inputShape))
        b = int(np.sqrt(self.outputShape))
        image = PIL.Image.fromarray(tile_raster_images(
                X=z,
                img_shape=(a, a), tile_shape=(b, b),
                tile_spacing=(1, 1)))
        image.save(fname)

    def __str__(self):
        return "Hidden Layer Activation %s; %d parameters"%(self.actFn,self.nParams)
    
class Softmax(Layer):
    def __init__(self, inputShape, nClasses):
        super(Softmax,self).__init__(inputShape, nClasses, T.nnet.softmax)

    def accuracy(self, y):
        p_y_given_x = self.getActivation()
        y_pred = T.argmax(p_y_given_x, axis=1)
        accuracy = T.mean(T.neq(y_pred, y))
        return accuracy

class Convolve(Layer):
    def __init__(self, inputShape, filterShape):
        """
            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height,filter width)
        """
        from theano.tensor.nnet import conv
        assert inputShape[1] == filterShape[1]
        self.filterShape      = filterShape
        self.inputShape       = inputShape
        # Output is (batchSize, nFilters, convolvedHeight, convolvedWidth)
        self.outputShape      = (inputShape[0], filterShape[0],
                                 inputShape[2]-filterShape[2]+1,
                                 inputShape[3]-filterShape[3]+1)
        self.nWeightParams    = np.prod(filterShape)
        self.nBiasParams      = filterShape[0]
        self.nParams          = self.nWeightParams + self.nBiasParams
        self.actFn            = conv.conv2d
    
    def getx0(self):
        fan_in = np.prod(self.filterShape[1:])
        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(3./fan_in),
              high=np.sqrt(3./fan_in),
              size=self.filterShape), dtype=theano.config.floatX)
        b_values = np.zeros((self.filterShape[0],), dtype=theano.config.floatX)
        return np.concatenate((W_values.flatten(),b_values))

    def setTheta(self, theta):
        self.W = theta[:np.prod(self.filterShape)].reshape(self.filterShape)
        self.b = theta[np.prod(self.filterShape):]

    def forward(self, x):
        convOut = self.actFn(x, self.W,
                             filter_shape=self.filterShape, image_shape=self.inputShape)
        self.act = convOut + self.b.dimshuffle('x',0,'x','x')
        return self.act

    def __str__(self):
        return "Convolution Layer: Filter %s; %d parameters"%(self.filterShape,self.nParams)

class Pool(Layer):
    def __init__(self, inputShape, poolsize=(2,2), ignoreborder=True):
        from theano.tensor.signal import downsample
        self.inputShape   = inputShape
        if ignoreborder:
            self.outputShape  = (inputShape[0], inputShape[1],
                                 int(inputShape[2] / poolsize[0]),
                                 int(inputShape[3] / poolsize[1]))
        else:
            self.outputShape  = (inputShape[0], inputShape[1],
                                 int(np.ceil(inputShape[2] / float(poolsize[0]))),
                                 int(np.ceil(inputShape[3] / float(poolsize[1]))))
        self.poolsize     = poolsize
        self.ignoreborder = ignoreborder
        self.actFn        = downsample.max_pool_2d
        self.nParams      = 0

    def getx0(self):
        return np.array([],dtype=theano.config.floatX)

    def setTheta(self, theta):
        pass
    
    def forward(self, x):
        ''' x must be a 4-tensor (image, channel, width, height) '''
        self.act = self.actFn(x, self.poolsize, ignore_border=self.ignoreborder)
        return self.act

    def __str__(self):
        return "Max Pooling Layer: poolsize %s"%(self.poolsize,)
        
class Reshape(Layer):
    def __init__(self, inputShape, outputShape):
        self.inputShape  = inputShape
        self.outputShape = outputShape
        self.nParams     = 0
        
    def getx0(self):
        return np.array([],dtype=theano.config.floatX)

    def setTheta(self, theta):
        pass
    
    def forward(self, x):
        self.act = x.reshape(self.outputShape)
        return self.act

    def __str__(self):
        return "Reshaping Layer"
