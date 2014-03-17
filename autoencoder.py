import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import dct
import mnist
from utils import tile_raster_images
import PIL.Image
import scipy.optimize
import argparse

rng = np.random

parser = argparse.ArgumentParser(description='Testing dct transforms')
parser.add_argument('--nStripes', required=False, type=int, default=6)
parser.add_argument('--stripeWidth', required=False, type=int, default=28)
parser.add_argument('--trainEpochs', required=False, type=int, default=200)
parser.add_argument('--outputPrefix', required=False, type=str, default='out')
parser.add_argument('--path', required=False, default='.')
args = parser.parse_args()

#=================== Parameters ===========================#
visibleSize  = 28*28             # Number of input units 
hiddenSize   = 14*14             # Number of hidden units 
alpha        = 9e-1              # Learning rate
Lambda       = 3e-3              # Weight decay term
beta         = 3                 # Weight of sparsity penalty term       
spar         = 0.1               # Sparsity parameter
path         = args.path         # Directory to load/save files
outputPrefix = args.outputPrefix # Prefix for output file names
nStripes     = args.nStripes     # Number of bands of coeffs to learn over
stripeWidth  = args.stripeWidth  # How wide each stripe is
trainEpochs  = args.trainEpochs  # How many epochs to train
useDCT       = nStripes > 0      # Enable dct compression

#================== Load the dataset ==========================#
images, labels = mnist.read(range(10),'training',path) 
images = images / 255.        # Remap between [0,1]
patches = images[:10000]      # Matlab patches use an odd col-major order
for i in range(len(patches)): # Convert them back over to row-major
    patches[i,:] = patches[i,:].reshape(28,28).T.flatten()
train_set_x = theano.shared(np.asarray(patches, dtype=theano.config.floatX))

nTrain        = train_set_x.shape[0].eval() # Number training samples
batch_size    = nTrain                      # Size of minibatches
nTrainBatches = nTrain / batch_size         # Number of minibatches

#================== Initialize Theano Vars ==========================#
nDctRows = int(stripeWidth * (nStripes-.5)) # Number of rows of dct coeffs we are keeping
nDctCoeffs = hiddenSize * nDctRows # Number of dct coefficients for each weight matrix
nWeightParams = 2*nDctCoeffs if useDCT else 2*visibleSize*hiddenSize
nBiasParams = visibleSize + hiddenSize
nParams = nWeightParams + nBiasParams
theta = theano.shared(value=np.zeros(nParams,dtype=theano.config.floatX),name='theta',borrow=True)

if useDCT:
    print "Performing DCT transform: %d stripes of width %d. %d total parameters."%(nStripes,stripeWidth,nParams)
    cW1 = theta[:nDctCoeffs].reshape((nDctRows, hiddenSize))
    cW2 = theta[nDctCoeffs:2*nDctCoeffs].reshape((nDctRows, hiddenSize))
    cb1 = theta[2*nDctCoeffs:2*nDctCoeffs+hiddenSize]
    cb2 = theta[2*nDctCoeffs+hiddenSize:]

    dctW1 = dct.dct((visibleSize, hiddenSize)) # Create the DCT transform
    dctW2 = dct.dct((hiddenSize, visibleSize))
    # dct_cb1 = dct.dct(cb1.shape.eval(), (hiddenSize,))
    # dct_cb2 = dct.dct(cb2.shape.eval(), (visibleSize,))

    E = np.zeros([visibleSize, nDctRows]) # Generate expansion matrix E
    sw = stripeWidth
    E[:sw/2,:sw/2] = np.identity(sw/2) # First stripe - half width
    for i in range(1,nStripes):
        E[i*sw*2-sw/2:i*sw*2+sw/2, sw/2+(i-1)*sw:sw/2+i*sw] = np.identity(sw)

    dW1 = T.dot(E, cW1)    # Expand coefficients
    dW2 = T.dot(E, cW2).T
    
    W1 = dctW1.idct2(dW1) # Inverse DCT transform
    W2 = dctW2.idct2(dW2)
    # b1 = dct_cb1.idct(cb1)
    # b2 = dct_cb2.idct(cb2)
    b1 = cb1
    b2 = cb2

else:
    print "Directly learning weights and biases (no DCT). %d total parameters." % nParams
    W1 = theta[:visibleSize*hiddenSize].reshape((visibleSize, hiddenSize))
    W2 = theta[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape((hiddenSize,visibleSize))
    b1 = theta[2*visibleSize*hiddenSize:2*visibleSize*hiddenSize+hiddenSize]
    b2 = theta[2*visibleSize*hiddenSize+hiddenSize:]
    
#================== Forward Propagate ==========================#
index = T.lscalar()      # Index into the batch of training examples
x = T.matrix('x')        # Training data 
a1 = T.nnet.sigmoid(T.dot(x, W1) + b1)
a2 = T.nnet.sigmoid(T.dot(a1, W2) + b2)

#================== Compute Cost ==========================#
m = x.shape[0]           # Number training examples
sse = T.sum((a2 - x) ** 2) / (2. * m)
avgAct = a1.mean(axis=0) # Col-Mean: AvgAct of each hidden unit across all m-examples
KL_Div = beta * T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))
weightDecayPenalty = (Lambda/2.) * (T.sum(W1**2) + T.sum(W2**2))
cost = sse + KL_Div + weightDecayPenalty

#================== Theano Functions ==========================#
batch_cost = theano.function( # Compute the cost of a minibatch
    inputs=[index],
    outputs=cost,
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_cost")

batch_grad = theano.function( # Compute the gradient of a minibatch
    inputs=[index],
    outputs=T.grad(cost, theta),
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_grad")

def trainFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in xrange(nTrainBatches)]
    meanLoss = np.mean(train_losses)
    return meanLoss

def gradFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    grad = batch_grad(0)
    for i in xrange(1, nTrainBatches):
        grad += batch_grad(i * batch_size)
    return grad / nTrainBatches

def callbackFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in xrange(nTrainBatches)]
    print 'Epoch',callbackFn.epoch,np.mean(train_losses)
    sys.stdout.flush()
    callbackFn.epoch += 1
callbackFn.epoch = 0

#================== Initialize Weights & Biases ==========================#
r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams))).astype('float32')
if useDCT: # Find coefficients that expand to the correct initial weight matrices
    dctShrink_cW1 = dct.dct((visibleSize, hiddenSize))
    iW1 = np.dot(E.T,dctShrink_cW1.dct2(rng.randn(visibleSize, hiddenSize)*2*r-r))
    dctShrink_cW2 = dct.dct((hiddenSize, visibleSize))
    iW2 = np.dot(E.T,dctShrink_cW2.dct2(rng.randn(hiddenSize, visibleSize)*2*r-r).T).T
    dctShrink_cb1 = dct.dct((hiddenSize,))
    ib1 = dctShrink_cb1.dct(np.zeros(hiddenSize))[:hiddenSize]
    dctShrink_cb2 = dct.dct((visibleSize,))
    ib2 = dctShrink_cb2.dct(np.zeros(visibleSize))[:visibleSize]
    x0 = np.concatenate([iW1.flatten(),iW2.flatten(),ib1,ib2]).astype('float32')

#================== Optimize ==========================#
start = time.time()
opttheta = scipy.optimize.fmin_cg(
    f=trainFn,
    x0=x0,
    fprime=gradFn,
    callback=callbackFn,
    maxiter=trainEpochs)
end = time.time()
print 'Elapsed Time(s): ', end - start

#================== Save W1 Image ==========================#
fname = path + '/results/' + outputPrefix + '.png'
theta.set_value(opttheta, borrow=True)
W = dctW1.idct2(T.dot(E,cW1)).eval().T if useDCT else W1.eval().T
image = PIL.Image.fromarray(tile_raster_images(
        X=W,
        img_shape=(28, 28), tile_shape=(14, 14),
        tile_spacing=(1, 1)))
image.save(fname)
