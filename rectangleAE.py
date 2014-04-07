import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dct
import mnist
from utils import tile_raster_images
import PIL.Image
import scipy.optimize
import argparse

rng = np.random

parser = argparse.ArgumentParser(description='Testing dct transforms')
parser.add_argument('--compression', required=False, type=float, default=.5)
parser.add_argument('--nEpochs', required=False, type=int, default=200)
parser.add_argument('--outputPrefix', required=False, type=str, default='out')
parser.add_argument('--path', required=False, default='.')
parser.add_argument('--noKLDiv', action='store_true', default=False)
parser.add_argument('--noWeightCost', action='store_true', default=False)
parser.add_argument('--noDCTWeightCost', action='store_true', default=False)
parser.add_argument('--dataDCT', action='store_true', default=False)
parser.add_argument('--dctLambda', required=False, type=float, default=3e-3)
args = parser.parse_args()

#=================== Parameters ===========================#
inputShape    = (28, 28)          # Dimensionality of input 
visibleSize   = 28*28             # Number of input units 
hiddenSize    = 14*14             # Number of hidden units 
alpha         = 9e-1              # Learning rate
Lambda        = 3e-3              # Weight decay term
dctLambda     = args.dctLambda    # DCT-weight decay term
beta          = 3                 # Weight of sparsity penalty term       
spar          = 0.1               # Sparsity parameter
compression   = args.compression  # Percentage compression
path          = args.path         # Directory to load/save files
outputPrefix  = args.outputPrefix # Prefix for output file names
trainEpochs   = args.nEpochs      # How many epochs to train
dataDCT       = args.dataDCT      # Performs DCT transform on the dataset
nWeightParams = 2*visibleSize*hiddenSize
nBiasParams   = visibleSize + hiddenSize
nParams       = nWeightParams + nBiasParams

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

#================== Initialize Theano Vars ==========================#
dctW1Shape = (int(np.round(np.sqrt(compression*visibleSize*visibleSize))),
              int(np.round(np.sqrt(compression*hiddenSize*hiddenSize))))
dctW2Shape = (dctW1Shape[1],dctW1Shape[0])
dctWeightSize = dctW1Shape[0]*dctW1Shape[1]
nDCTParams = 2*dctWeightSize + nBiasParams

theta = theano.shared(value=np.zeros(nDCTParams,dtype=theano.config.floatX),name='theta',borrow=True)

print "Learning in DCT space\n%d total parameters (%f%%)\nDCT input shape"\
    %(nDCTParams,100.*nDCTParams/nParams), dctW1Shape

cW1 = theta[:dctWeightSize].reshape(dctW1Shape)
cW2 = theta[dctWeightSize:2*dctWeightSize].reshape(dctW2Shape)
b1 = theta[2*dctWeightSize:2*dctWeightSize+hiddenSize]
b2 = theta[2*dctWeightSize+hiddenSize:]

d1 = dct.dct(dctW1Shape,(visibleSize,hiddenSize))
d2 = dct.dct(dctW2Shape,(hiddenSize,visibleSize))

W1 = d1.idct2(cW1)
W2 = d2.idct2(cW2)
    
#================== Forward Propagate ==========================#
index = T.lscalar()      # Index into the batch of training examples
x = T.matrix('x')        # Training data 
a1 = T.nnet.sigmoid(T.dot(x, W1) + b1)
a2 = T.nnet.sigmoid(T.dot(a1, W2) + b2)

#================== Compute Cost ==========================#
m = x.shape[0]           # Number training examples
cost = T.sum((a2 - x) ** 2) / (2. * m) # Sum of squared errors
# TODO: Consider Cross Entropy Loss
if not args.noKLDiv:
    print 'Using KL-Divergence Cost. Gain:', beta
    avgAct = a1.mean(axis=0) # Col-Mean: AvgAct of each hidden unit across all m-examples
    KL_Div = beta * T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))
    cost += KL_Div
if not args.noWeightCost:
    print 'Using Standard Weight Penalty. Gain:', Lambda
    weightDecayPenalty = (Lambda/2.) * (T.sum(W1**2) + T.sum(W2**2))
    cost += weightDecayPenalty
if not args.noDCTWeightCost:
    print 'Using DCT-Weight Penalty. Gain:', dctLambda
    # dctWeightDecayPenalty = (dctLambda/2.) * (T.sum(cW1**2) + T.sum(cW2**2))
    # cost += dctWeightDecayPenalty
    pdf = np.vectorize(scipy.stats.norm().pdf)

    w1tmp = np.outer(pdf(np.linspace(0,2,dctW1Shape[0])),pdf(np.linspace(0,2,dctW1Shape[1])))
    cW1Penalty = 1.-(w1tmp/np.max(w1tmp))
    
    w2tmp = np.outer(pdf(np.linspace(0,2,dctW2Shape[0])),pdf(np.linspace(0,2,dctW2Shape[1])))
    cW2Penalty = 1.-(w2tmp/np.max(w2tmp))
    
    penW1 = theano.shared(value=cW1Penalty.astype('float32'),borrow=True)    
    penW2 = theano.shared(value=cW2Penalty.astype('float32'),borrow=True)

    dctWeightDecayPenalty = (dctLambda/2.) * (T.sum(penW1 * (cW1**2)) + T.sum(penW2 * (cW2**2)))
    cost += dctWeightDecayPenalty


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
    train_losses = [batch_cost(i) for i in xrange(nTrainBatches)]
    test_losses = [test_cost(i) for i in xrange(nTestBatches)]
    print 'Epoch %d Train: %f Test: %f'%(callbackFn.epoch,np.mean(train_losses),np.mean(test_losses))
    sys.stdout.flush()
    callbackFn.epoch += 1
callbackFn.epoch = 0

#================== Initialize Weights & Biases ==========================#
r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),np.zeros(nBiasParams))).astype('float32')
x0w1 = x0[:visibleSize*hiddenSize].reshape((visibleSize, hiddenSize))
dctShrink_cW1 = dct.dct((visibleSize, hiddenSize))
iW1 = dctShrink_cW1.dct2(x0w1)[:dctW1Shape[0],:dctW1Shape[1]]
x0w2 = x0[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape((hiddenSize,visibleSize))
dctShrink_cW2 = dct.dct((hiddenSize, visibleSize))
iW2 = dctShrink_cW2.dct2(x0w2)[:dctW2Shape[0],:dctW2Shape[1]]
x0 = np.concatenate([iW1.flatten(),iW2.flatten(),np.zeros(nBiasParams)]).astype('float32')

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
image = PIL.Image.fromarray(tile_raster_images(
        X=W1.eval().T,
        img_shape=(28, 28), tile_shape=(14, 14),
        tile_spacing=(1, 1)))
image.save(fname)
