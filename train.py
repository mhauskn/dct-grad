import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mnist
from utils import tile_raster_images
import PIL.Image
import scipy.optimize
import argparse
from autoencoder import *

parser = argparse.ArgumentParser(description='Testing dct transforms')
parser.add_argument('--autoencoder', required=False, type=str, default='autoencoder')
parser.add_argument('--compression', required=False, type=float, default=.5)
parser.add_argument('--nEpochs', required=False, type=int, default=200)
parser.add_argument('--outputPrefix', required=False, type=str, default='out')
parser.add_argument('--path', required=False, default='.')
parser.add_argument('--dataDCT', action='store_true', default=False)
parser.add_argument('--beta', required=False, type=float, default=3)
parser.add_argument('--spar', required=False, type=float, default=.1)
parser.add_argument('--Lambda', required=False, type=float, default=3e-3)
args = parser.parse_args()

#=================== Parameters ===========================#
inputShape    = (28, 28)             # Dimensionality of input 
visibleSize   = 28*28                # Number of input units 
hiddenSize    = 14*14                # Number of hidden units 
Lambda        = args.Lambda          # Weight decay term
beta          = args.beta            # Weight of sparsity penalty term       
spar          = args.spar            # Sparsity parameter
compression   = args.compression     # Percentage compression
path          = args.path            # Directory to load/save files
outputPrefix  = args.outputPrefix    # Prefix for output file names
trainEpochs   = args.nEpochs         # How many epochs to train
dataDCT       = args.dataDCT         # Performs DCT transform on the dataset
aeType        = args.autoencoder     # What type of autoencoder

#================== Load the dataset ==========================#
if dataDCT: print 'Applying 2d-DCT transform to the dataset.'
images = mnist.read(range(10),'training',path, dataDCT)[0]
# plt.imshow(images[0,:].reshape(28,28), cmap=cm.Greys_r)
train_set_x = theano.shared(np.asarray(images[:10000], dtype=theano.config.floatX))
images = mnist.read(range(10),'testing',path, dataDCT)[0]
test_set_x = theano.shared(np.asarray(images, dtype=theano.config.floatX))

nTrain        = train_set_x.shape[0].eval() # Number training samples
nTest         = test_set_x.shape[0].eval()  # Number of test samples
batch_size    = nTrain                      # Size of minibatches
nTrainBatches = nTrain / batch_size         # Number of minibatches
nTestBatches  = nTest / batch_size          

if aeType == 'autoencoder': ae = Autoencoder(visibleSize, hiddenSize)
elif aeType == 'rectangle': ae = RectangleAE(visibleSize, hiddenSize)
elif aeType == 'stripe':    ae = StripeAE(visibleSize, hiddenSize)
elif aeType == 'reshape':   ae = ReshapeAE(visibleSize, hiddenSize, inputShape)
else: assert(False)

index = T.lscalar()      # Index into the batch of training examples
x = T.matrix('x')        # Training data 
a1, a2 = ae.forward(x)   # Forward Propagate
cost = reconstructionCost(x,a2) + beta * sparsityCost(a1,spar) + Lambda * ae.weightCost()

#================== Theano Functions ==========================#
batch_cost = theano.function( # Compute the cost of a minibatch
    inputs=[index],
    outputs=cost,
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_cost")

test_cost = theano.function( # Compute the cost of a minibatch
    inputs=[index],
    outputs=cost,
    givens={x:test_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_cost")

batch_grad = theano.function( # Compute the gradient of a minibatch
    inputs=[index],
    outputs=T.grad(cost, ae.theta),
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_grad")

def trainFn(theta_value):
    ae.theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in xrange(nTrainBatches)]
    meanLoss = np.mean(train_losses)
    return meanLoss

def gradFn(theta_value):
    ae.theta.set_value(theta_value, borrow=True)
    grad = batch_grad(0)
    for i in xrange(1, nTrainBatches):
        grad += batch_grad(i * batch_size)
    return grad / nTrainBatches

def callbackFn(theta_value):
    ae.theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i) for i in xrange(nTrainBatches)]
    test_losses = [test_cost(i) for i in xrange(nTestBatches)]
    print 'Epoch %d Train: %f Test: %f'%(callbackFn.epoch,np.mean(train_losses),np.mean(test_losses))
    sys.stdout.flush()
    callbackFn.epoch += 1
callbackFn.epoch = 0

#================== Optimize ==========================#
start = time.time()
opttheta = scipy.optimize.fmin_cg(
    f=trainFn,
    x0=ae.x0,
    fprime=gradFn,
    callback=callbackFn,
    maxiter=trainEpochs)
end = time.time()
print 'Elapsed Time(s): ', end - start

ae.saveImage(outputPrefix, opttheta)
