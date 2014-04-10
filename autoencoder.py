import numpy as np
import theano
import theano.tensor as T
import dct
import PIL.Image
from utils import tile_raster_images

rng = np.random

def sparsityCost(a1, spar):
    avgAct = a1.mean(axis=0) # Col-Mean: AvgAct of each hidden unit across all m-examples
    return T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))

def reconstructionCost(x, a):
    return T.sum((a - x) ** 2) / (2. * x.shape[0])

class Autoencoder():
    def __init__(self, visibleSize, hiddenSize):
        nWeightParams = 2*visibleSize*hiddenSize
        nBiasParams   = visibleSize + hiddenSize
        nParams       = nWeightParams + nBiasParams
        print "Direct Autoencoder\n%d total parameters"%nParams

        self.theta = theano.shared(value=np.zeros(nParams,dtype=theano.config.floatX),
                                   name='theta',borrow=True)
        self.W1 = self.theta[:visibleSize*hiddenSize].reshape((visibleSize, hiddenSize))
        self.W2 = self.theta[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape((hiddenSize,visibleSize))
        self.b1 = self.theta[2*visibleSize*hiddenSize:2*visibleSize*hiddenSize+hiddenSize]
        self.b2 = self.theta[2*visibleSize*hiddenSize+hiddenSize:]

        r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
        self.x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams))).astype('float32')

    def forward(self, x):
        a1 = T.nnet.sigmoid(T.dot(x, self.W1) + self.b1)
        a2 = T.nnet.sigmoid(T.dot(a1, self.W2) + self.b2)
        return a1, a2

    def weightCost(self):
        return T.sum(self.W1**2) + T.sum(self.W2**2) # L2
        # return T.sum(T.abs_(self.W1)) + T.sum(T.abs_(self.W2)) # L1
        
    def saveImage(self, fname, opttheta):
        self.theta.set_value(opttheta, borrow=True)
        image = PIL.Image.fromarray(tile_raster_images(
                X=self.W1.eval().T,
                img_shape=(28, 28), tile_shape=(14, 14),
                tile_spacing=(1, 1)))
        image.save(fname)
        

class RectangleAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, compression=.5):
        nWeightParams = 2*visibleSize*hiddenSize
        nBiasParams   = visibleSize + hiddenSize
        nParams = nWeightParams + nBiasParams

        self.dctW1Shape = (int(np.round(np.sqrt(compression*visibleSize*visibleSize))),
                           int(np.round(np.sqrt(compression*hiddenSize*hiddenSize))))
        self.dctW2Shape = (self.dctW1Shape[1],self.dctW1Shape[0])
        self.dctWeightSize = self.dctW1Shape[0]*self.dctW1Shape[1]
        self.nDCTParams = 2 * self.dctWeightSize + nBiasParams

        print "RectangleAE learning in DCT space\n%d total parameters (%f%%)\nDCT input shape"\
            %(self.nDCTParams,100.*self.nDCTParams/nParams), self.dctW1Shape

        self.theta = theano.shared(value=np.zeros(self.nDCTParams,dtype=theano.config.floatX),name='theta',borrow=True)

        self.cW1 = self.theta[:self.dctWeightSize].reshape(self.dctW1Shape)
        self.cW2 = self.theta[self.dctWeightSize:2*self.dctWeightSize].reshape(self.dctW2Shape)
        self.b1 = self.theta[2*self.dctWeightSize:2*self.dctWeightSize+hiddenSize]
        self.b2 = self.theta[2*self.dctWeightSize+hiddenSize:]

        self.d1 = dct.dct(self.dctW1Shape,(visibleSize,hiddenSize))
        self.d2 = dct.dct(self.dctW2Shape,(hiddenSize,visibleSize))

        self.W1 = self.d1.idct2(self.cW1)
        self.W2 = self.d2.idct2(self.cW2)
    
        r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
        x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams))).astype('float32')
        x0w1 = x0[:visibleSize*hiddenSize].reshape((visibleSize, hiddenSize))
        dctShrink_cW1 = dct.dct((visibleSize, hiddenSize))
        iW1 = dctShrink_cW1.dct2(x0w1)[:self.dctW1Shape[0],:self.dctW1Shape[1]]
        x0w2 = x0[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape((hiddenSize,visibleSize))
        dctShrink_cW2 = dct.dct((hiddenSize, visibleSize))
        iW2 = dctShrink_cW2.dct2(x0w2)[:self.dctW2Shape[0],:self.dctW2Shape[1]]
        self.x0 = np.concatenate([iW1.flatten(),iW2.flatten(),np.zeros(nBiasParams)]).astype('float32')

    def weightCost(self):
        return T.sum(self.cW1**2) + T.sum(self.cW2**2)

        # pdf = np.vectorize(scipy.stats.norm().pdf)
        # w1tmp = np.outer(pdf(np.linspace(0,2,dctW1Shape[0])),pdf(np.linspace(0,2,dctW1Shape[1])))
        # cW1Penalty = 1.-(w1tmp/np.max(w1tmp))
        # w2tmp = np.outer(pdf(np.linspace(0,2,dctW2Shape[0])),pdf(np.linspace(0,2,dctW2Shape[1])))
        # cW2Penalty = 1.-(w2tmp/np.max(w2tmp))
        # penW1 = theano.shared(value=cW1Penalty.astype('float32'),borrow=True)    
        # penW2 = theano.shared(value=cW2Penalty.astype('float32'),borrow=True)
        # return T.sum(penW1 * T.abs_(cW1)) + T.sum(penW2 * T.abs_(cW2))
        
class StripeAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, nStripes=9):
        nWeightParams = 2*visibleSize*hiddenSize
        nBiasParams = visibleSize + hiddenSize
        nParams = nWeightParams + nBiasParams
        stripeWidth = int(np.round(np.sqrt(visibleSize)))
        nDctRows = int(stripeWidth * (nStripes-.5)) # Number of rows of dct coeffs we are keeping
        nDctCoeffs = hiddenSize * nDctRows # Number of dct coefficients for each weight matrix
        nDCTWeightParams = 2*nDctCoeffs 
        nDCTParams = nDCTWeightParams + nBiasParams

        print "StripeAE learning in DCT Space\n%d stripes of width %d\n%d total parameters (%f%%)"\
            %(nStripes,stripeWidth,nDCTParams,100.*nDCTParams/nParams)

        self.theta = theano.shared(value=np.zeros(nDCTParams,dtype=theano.config.floatX),name='theta',borrow=True)

        self.cW1 = self.theta[:nDctCoeffs].reshape((nDctRows, hiddenSize))
        self.cW2 = self.theta[nDctCoeffs:2*nDctCoeffs].reshape((nDctRows, hiddenSize))
        self.b1 = self.theta[2*nDctCoeffs:2*nDctCoeffs+hiddenSize]
        self.b2 = self.theta[2*nDctCoeffs+hiddenSize:]

        self.dctW1 = dct.dct((visibleSize, hiddenSize)) # Create the DCT transform
        self.dctW2 = dct.dct((hiddenSize, visibleSize))

        E = np.zeros([visibleSize, nDctRows]) # Generate expansion matrix E
        sw = stripeWidth
        E[:sw/2,:sw/2] = np.identity(sw/2) # First stripe - half width
        for i in range(1,nStripes):
            E[i*sw*2-sw/2:i*sw*2+sw/2, sw/2+(i-1)*sw:sw/2+i*sw] = np.identity(sw)

        self.sE = theano.shared(value=E.astype('float32'),name='E',borrow=True)

        self.dW1 = T.dot(self.sE, self.cW1)   # Expand coefficients
        self.dW2 = T.dot(self.sE, self.cW2).T

        self.W1 = self.dctW1.idct2(self.dW1) # Inverse DCT transform
        self.W2 = self.dctW2.idct2(self.dW2)

        r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
        dctShrink_cW1 = dct.dct((visibleSize, hiddenSize))
        iW1 = np.dot(E.T,dctShrink_cW1.dct2(rng.randn(visibleSize, hiddenSize)*2*r-r))
        dctShrink_cW2 = dct.dct((hiddenSize, visibleSize))
        iW2 = np.dot(E.T,dctShrink_cW2.dct2(rng.randn(hiddenSize, visibleSize)*2*r-r).T).T
        self.x0 = np.concatenate([iW1.flatten(),iW2.flatten(),np.zeros(nBiasParams)]).astype('float32')

    def weightCost(self):
        return T.sum(self.cW1**2) + T.sum(self.cW2**2)
        # pdf = np.vectorize(scipy.stats.norm().pdf)
        # wtmp = np.outer(pdf(np.linspace(0,2,nDctRows)),pdf(np.linspace(0,2,hiddenSize)))
        # cWPenalty = 1.-(wtmp/np.max(wtmp))
        # penW = theano.shared(value=cWPenalty.astype('float32'),borrow=True)    
        # return T.sum(penW * T.abs_(cW1)) + T.sum(penW * T.abs_(cW2))

class ReshapeAE(Autoencoder):
    def __init__(self, visibleSize, hiddenSize, inputShape, compression=.5):    
        nWeightParams = 2*visibleSize*hiddenSize
        nBiasParams = visibleSize + hiddenSize
        nParams = nWeightParams + nBiasParams
        dctShape = (int(np.round(np.sqrt(inputShape[0]*compression*visibleSize/float(inputShape[1])))),
                    int(np.round(np.sqrt(inputShape[1]*compression*visibleSize/float(inputShape[0])))))
        dctVisibleSize = dctShape[0]*dctShape[1]
        nDCTParams = 2*dctVisibleSize*hiddenSize + nBiasParams

        print "ReshapeAE learning in DCT space\n%d total parameters (%f%%)\nDCT input shape"\
            %(nDCTParams,100.*nDCTParams/nParams), dctShape


        self.theta = theano.shared(value=np.zeros(nDCTParams,dtype=theano.config.floatX),name='theta',borrow=True)

        self.cW1 = self.theta[:dctVisibleSize*hiddenSize].reshape((dctVisibleSize, hiddenSize))
        self.cW2 = self.theta[dctVisibleSize*hiddenSize:2*dctVisibleSize*hiddenSize].reshape((hiddenSize,dctVisibleSize))
        self.b1 = self.theta[2*dctVisibleSize*hiddenSize:2*dctVisibleSize*hiddenSize+hiddenSize]
        self.b2 = self.theta[2*dctVisibleSize*hiddenSize+hiddenSize:]

        self.d = dct.dct(dctShape,inputShape) # Create the DCT transform

        x = T.matrix('x')
        self.W1, _ = theano.scan(fn=lambda x: self.d.idct2(x.reshape(dctShape)).flatten(),
                                 outputs_info=None,
                                 sequences=[self.cW1.T])
        self.W1 = self.W1.T

        self.W2, _ = theano.scan(fn=lambda x: self.d.idct2(x.reshape(dctShape)).flatten(),
                                 outputs_info=None,
                                 sequences=[self.cW2])

        r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
        x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams)))
        mydct = dct.dct(inputShape)
        x0w1 = x0[:visibleSize*hiddenSize].reshape((visibleSize, hiddenSize))
        x0w2 = x0[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape((hiddenSize,visibleSize))
        tmpW1 = np.vstack([mydct.dct2(x0w1[:,i].reshape(inputShape))[:dctShape[0],:dctShape[1]].flatten() for i in xrange(hiddenSize)]).T
        tmpW2 = np.vstack([mydct.dct2(x0w2[i].reshape(inputShape))[:dctShape[0],:dctShape[1]].flatten() for i in xrange(hiddenSize)])
        self.x0 = np.concatenate([tmpW1.flatten(),tmpW2.flatten(),np.zeros(nBiasParams)]).astype('float32')

        def weightCost(self):
            # L2-Regularization
            return T.sum(self.cW1**2) + T.sum(self.cW2**2)
            # L1-Regularization
            # return T.sum(T.abs_(cW1)) + T.sum(T.abs_(cW2))

            # Frequency weighted L2-Regularization
            # pdf = np.vectorize(scipy.stats.norm().pdf)
            # # Create the penalty gaussian for the dct-image
            # wtmp = np.outer(pdf(np.linspace(0,2,dctShape[0])),pdf(np.linspace(0,2,dctShape[1])))
            # imgPenalty = 1.-(wtmp/np.max(wtmp))
            # # Flatten and tile this into a matrix of size [dctVisibleSize, hiddenSize]
            # cWPenalty = np.tile(imgPenalty.flatten(),(hiddenSize,1)).T
            # penW = theano.shared(value=cWPenalty.astype('float32'),borrow=True)    
            # return T.sum(penW * (cW1**2)) + T.sum(penW.T * (cW2**2))
