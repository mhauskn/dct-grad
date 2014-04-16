import numpy as np
import theano
import theano.tensor as T
import dct
import scipy.stats
import PIL.Image
from utils import tile_raster_images

rng = np.random

def sparsityCost(a1, spar):
    avgAct = a1.mean(axis=0) # Col-Mean: AvgAct of each hidden unit across all m-examples
    return T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))

def reconstructionCost(x, a):
    return T.sum((a - x) ** 2) / (2. * x.shape[0])

def testReconstruction(data, autoencoder):
    dataAct = autoencoder.forward(data)
    dataCost = reconstructionCost(data, dataAct).eval()
    bogus = rng.random(data.shape)
    bogusAct = autoencoder.forward(bogus)
    bogusCost = reconstructionCost(bogus, bogusAct).eval()
    print 'Data Reconstruction Cost:', dataCost
    print 'Random Reconstruction Cost:', bogusCost

class Autoencoder(object):
    def __init__(self, visibleSize, hiddenSize, beta=3, spar=.1, Lambda=3e-3):
        self.visibleSize   = visibleSize
        self.hiddenSize    = hiddenSize
        self.beta          = beta
        self.spar          = spar
        self.Lambda        = Lambda
        self.nWeightParams = 2*visibleSize*hiddenSize
        self.nBiasParams   = visibleSize + hiddenSize
        self.nParams       = self.nWeightParams + self.nBiasParams

    def getNParams(self):
        return self.nParams

    def getx0(self):
        r = np.sqrt(6) / np.sqrt(self.visibleSize+self.hiddenSize+1)
        return np.concatenate(((rng.randn(self.nWeightParams)*2*r-r).flatten(),
                               np.zeros(self.nBiasParams)))

    def setTheta(self, theta):
        v = self.visibleSize
        h = self.hiddenSize
        self.W1 = theta[:v*h].reshape((v, h))
        self.W2 = theta[v*h:2*v*h].reshape((h,v))
        self.b1 = theta[2*v*h:2*v*h+h]
        self.b2 = theta[2*v*h+h:]

    def forward(self, x):
        self.a1 = T.nnet.sigmoid(T.dot(x, self.W1) + self.b1)
        a2 = T.nnet.sigmoid(T.dot(self.a1, self.W2) + self.b2)
        return a2

    def cost(self, x, output, labels):
        return reconstructionCost(x,output) + self.beta * sparsityCost(self.a1,self.spar) + self.Lambda * self.l2WeightCost()

    def l2WeightCost(self):
        return T.sum(self.W1**2) + T.sum(self.W2**2)

    def l1WeightCost(self):
        return T.sum(T.abs_(self.W1)) + T.sum(T.abs_(self.W2))
        
    def saveImage(self, fname):
        image = PIL.Image.fromarray(tile_raster_images(
                X=self.W1.eval().T,
                img_shape=(28, 28), tile_shape=(14, 14),
                tile_spacing=(1, 1)))
        image.save(fname)
        
    def __str__(self):
        return "Direct Autoencoder. %d parameters"%self.nParams


class RectangleAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, compression=.5, beta=3, spar=.1, Lambda=3e-3):
        super(RectangleAE,self).__init__(visibleSize, hiddenSize, beta, spar, Lambda)
        self.dctW1Shape = (int(np.round(np.sqrt(compression*visibleSize*visibleSize))),
                           int(np.round(np.sqrt(compression*hiddenSize*hiddenSize))))
        self.dctW2Shape = (self.dctW1Shape[1],self.dctW1Shape[0])
        self.dctWeightSize = self.dctW1Shape[0]*self.dctW1Shape[1]
        self.nDCTParams = 2 * self.dctWeightSize + self.nBiasParams

        self.d1 = dct.dct(self.dctW1Shape,(self.visibleSize,self.hiddenSize))
        self.d2 = dct.dct(self.dctW2Shape,(self.hiddenSize,self.visibleSize))

    def getNParams(self):
        return self.nDCTParams

    def getx0(self):
        x0 = super(RectangleAE,self).getx0()
        x0w1 = x0[:self.visibleSize*self.hiddenSize].reshape((self.visibleSize, self.hiddenSize))
        dctShrink_cW1 = dct.dct((self.visibleSize, self.hiddenSize))
        iW1 = dctShrink_cW1.dct2(x0w1)[:self.dctW1Shape[0],:self.dctW1Shape[1]]
        x0w2 = x0[self.visibleSize*self.hiddenSize:2*self.visibleSize*self.hiddenSize].reshape((self.hiddenSize,self.visibleSize))
        dctShrink_cW2 = dct.dct((self.hiddenSize, self.visibleSize))
        iW2 = dctShrink_cW2.dct2(x0w2)[:self.dctW2Shape[0],:self.dctW2Shape[1]]
        return np.concatenate([iW1.flatten(),iW2.flatten(),np.zeros(self.nBiasParams)])

    def setTheta(self, theta):
        self.cW1 = theta[:self.dctWeightSize].reshape(self.dctW1Shape)
        self.cW2 = theta[self.dctWeightSize:2*self.dctWeightSize].reshape(self.dctW2Shape)
        self.b1 = theta[2*self.dctWeightSize:2*self.dctWeightSize+self.hiddenSize]
        self.b2 = theta[2*self.dctWeightSize+self.hiddenSize:]
        self.W1 = self.d1.idct2(self.cW1)
        self.W2 = self.d2.idct2(self.cW2)

    def l2WeightCost(self):
        return T.sum(self.cW1**2) + T.sum(self.cW2**2)

    def l1WeightCost(self):
        return T.sum(T.abs_(self.cW1)) + T.sum(T.abs_(self.cW2))

    def freqWeightedL2Cost(self):
        pdf = np.vectorize(scipy.stats.norm().pdf)
        w1tmp = np.outer(pdf(np.linspace(0,2,self.dctW1Shape[0])),pdf(np.linspace(0,2,self.dctW1Shape[1])))
        cW1Penalty = 1.-(w1tmp/np.max(w1tmp))
        w2tmp = np.outer(pdf(np.linspace(0,2,self.dctW2Shape[0])),pdf(np.linspace(0,2,self.dctW2Shape[1])))
        cW2Penalty = 1.-(w2tmp/np.max(w2tmp))
        penW1 = theano.shared(value=cW1Penalty.astype('float32'),borrow=True) #TODO: Declare earlier
        penW2 = theano.shared(value=cW2Penalty.astype('float32'),borrow=True) #TODO: Declare earlier
        return T.sum(penW1 * (self.cW1**2)) + T.sum(penW2 * (self.cW2**2))
        
    def __str__(self):
        return "RectangleAE learning in DCT space. %d parameters"%(self.nDCTParams)

class StripeAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, nStripes=9, beta=3, spar=.1, Lambda=3e-3):
        super(StripeAE,self).__init__(visibleSize, hiddenSize, beta, spar, Lambda)
        self.nStripes = nStripes
        self.stripeWidth = int(np.round(np.sqrt(visibleSize)))
        # Number of rows of dct coeffs we are keeping
        self.nDctRows = int(self.stripeWidth * (self.nStripes-.5)) 
        # Number of dct coefficients for each weight matrix
        self.nDctCoeffs = hiddenSize * self.nDctRows 
        self.nDCTWeightParams = 2*self.nDctCoeffs 
        self.nDCTParams = self.nDCTWeightParams + self.nBiasParams

        self.E = np.zeros([visibleSize, self.nDctRows]) # Generate expansion matrix E
        sw = self.stripeWidth
        self.E[:sw/2,:sw/2] = np.identity(sw/2) # First stripe - half width
        for i in range(1,self.nStripes):
            self.E[i*sw*2-sw/2:i*sw*2+sw/2, sw/2+(i-1)*sw:sw/2+i*sw] = np.identity(sw)
        self.sE = theano.shared(value=self.E.astype('float32'),name='E',borrow=True)

    def getNParams(self):
        return self.nDCTParams

    def getx0(self):
        r = np.sqrt(6) / np.sqrt(self.visibleSize+self.hiddenSize+1)
        dctShrink_cW1 = dct.dct((self.visibleSize, self.hiddenSize))
        iW1 = np.dot(self.E.T,dctShrink_cW1.dct2(
                rng.randn(self.visibleSize, self.hiddenSize)*2*r-r))
        dctShrink_cW2 = dct.dct((self.hiddenSize, self.visibleSize))
        iW2 = np.dot(self.E.T,dctShrink_cW2.dct2(
                rng.randn(self.hiddenSize, self.visibleSize)*2*r-r).T).T
        return np.concatenate([iW1.flatten(),iW2.flatten(),np.zeros(self.nBiasParams)])

    def setTheta(self, theta):
        self.cW1 = theta[:self.nDctCoeffs].reshape((self.nDctRows, self.hiddenSize))
        self.cW2 = theta[self.nDctCoeffs:2*self.nDctCoeffs].reshape((self.nDctRows, self.hiddenSize))
        self.b1 = theta[2*self.nDctCoeffs:2*self.nDctCoeffs+self.hiddenSize]
        self.b2 = theta[2*self.nDctCoeffs+self.hiddenSize:]

        self.dctW1 = dct.dct((self.visibleSize, self.hiddenSize)) # Create the DCT transform
        self.dctW2 = dct.dct((self.hiddenSize, self.visibleSize))

        self.dW1 = T.dot(self.sE, self.cW1)   # Expand coefficients
        self.dW2 = T.dot(self.sE, self.cW2).T

        self.W1 = self.dctW1.idct2(self.dW1) # Inverse DCT transform
        self.W2 = self.dctW2.idct2(self.dW2)

    def l2WeightCost(self):
        return T.sum(self.cW1**2) + T.sum(self.cW2**2)

    def l1WeightCost(self):
        return T.sum(T.abs_(self.cW1)) + T.sum(T.abs_(self.cW2))

    def freqWeightedL2Cost(self):
        pdf = np.vectorize(scipy.stats.norm().pdf)
        wtmp = np.outer(pdf(np.linspace(0,2,self.nDctRows)),pdf(np.linspace(0,2,self.hiddenSize)))
        cWPenalty = 1.-(wtmp/np.max(wtmp))
        penW = theano.shared(value=cWPenalty.astype('float32'),borrow=True) #TODO: Performance
        return T.sum(penW * (self.cW1**2)) + T.sum(penW * (self.cW2**2))

    def __str__(self):
        return "StripeAE learning in DCT Space. %d stripes of width %d. %d parameters"\
            %(self.nStripes,self.stripeWidth,self.nDCTParams)


class ReshapeAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, inputShape,
                 compression=.5, beta=3, spar=.1, Lambda=3e-3):
        super(ReshapeAE,self).__init__(visibleSize, hiddenSize, beta, spar, Lambda)
        self.inputShape = inputShape
        self.dctShape = (int(np.round(np.sqrt(inputShape[0]*compression*visibleSize/
                                              float(inputShape[1])))),
                         int(np.round(np.sqrt(inputShape[1]*compression*visibleSize/
                                              float(inputShape[0])))))
        self.dctVisibleSize = self.dctShape[0]*self.dctShape[1]
        self.nDCTParams = 2*self.dctVisibleSize*hiddenSize + self.nBiasParams
        self.d = dct.dct(self.dctShape,self.inputShape) # Create the DCT transform

    def getNParams(self):
        return self.nDCTParams

    def getx0(self):
        v = self.visibleSize
        h = self.hiddenSize
        x0 = super(ReshapeAE,self).getx0()
        x0w1 = x0[:v*h].reshape((v, h))
        x0w2 = x0[v*h:2*v*h].reshape((h,v))
        mydct = dct.dct(self.inputShape)
        tmpW1 = np.vstack([mydct.dct2(x0w1[:,i].reshape(self.inputShape))\
                               [:self.dctShape[0],:self.dctShape[1]].flatten()\
                               for i in xrange(h)]).T
        tmpW2 = np.vstack([mydct.dct2(x0w2[i].reshape(self.inputShape))\
                               [:self.dctShape[0],:self.dctShape[1]].flatten()\
                               for i in xrange(h)])
        return np.concatenate([tmpW1.flatten(),tmpW2.flatten(),np.zeros(self.nBiasParams)])

    def setTheta(self, theta):
        dv = self.dctVisibleSize
        h = self.hiddenSize

        n = 0
        self.cW1 = theta[n:n+dv*h].reshape((dv, h))
        n += dv*h
        self.cW2 = theta[n:n+dv*h].reshape((h, dv))
        n += dv*h
        self.b1 = theta[n:n+h]
        n += h
        self.b2 = theta[n:]

        x = T.matrix('x')
        self.W1, _ = theano.scan(fn=lambda x: self.d.idct2(x.reshape(self.dctShape)).flatten(),
                                 outputs_info=None,
                                 sequences=[self.cW1.T])
        self.W1 = self.W1.T

        self.W2, _ = theano.scan(fn=lambda x: self.d.idct2(x.reshape(self.dctShape)).flatten(),
                                 outputs_info=None,
                                 sequences=[self.cW2])

    def l2WeightCost(self):
        return T.sum(self.cW1**2) + T.sum(self.cW2**2)

    def l1WeightCost(self):
        return T.sum(T.abs_(self.cW1)) + T.sum(T.abs_(self.cW2))

    def freqWeightedL2Cost(self):
        pdf = np.vectorize(scipy.stats.norm().pdf)
        # Create the penalty gaussian for the dct-image
        wtmp = np.outer(pdf(np.linspace(0,2,self.dctShape[0])),pdf(np.linspace(0,2,self.dctShape[1])))
        imgPenalty = 1.-(wtmp/np.max(wtmp))
        # Flatten and tile this into a matrix of size [dctVisibleSize, hiddenSize]
        cWPenalty = np.tile(imgPenalty.flatten(),(self.hiddenSize,1)).T
        penW = theano.shared(value=cWPenalty.astype('float32'),borrow=True) #TODO: Remove?
        return T.sum(penW * (self.cW1**2)) + T.sum(penW.T * (self.cW2**2))

    def __str__(self):
        return "ReshapeAE learning in DCT space. %d parameters"%(self.nDCTParams)
            
