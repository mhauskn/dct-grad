import time
import sys
from itertools import *
import numpy as np
import theano
import theano.tensor as T
import mnist
import scipy.io
import argparse

rng = np.random

parser = argparse.ArgumentParser(description='Softmax Regression')
parser.add_argument('--weights', required=False, type=str, default='weights.mat')
parser.add_argument('--path', required=False, default='.')
parser.add_argument('--nEpochs', required=False, type=int, default=200)
args = parser.parse_args()

#=================== Parameters ===========================#
inputSize   = 28*28                  # Number of input units 
nClasses    = 10;                    # Number of classes (MNIST images fall into 10 classes)
lambdaP     = 1e-4;                  # Weight decay parameter
weightFile  = args.weights           # Weight file to load
path        = args.path              # Directory to load/save files
nEpochs     = args.nEpochs           # How many epochs to train
nParams     = nClasses*(inputSize+1) # Number of paramters in the model

#================== Load the dataset ==========================#
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    data_y = np.fromiter(chain.from_iterable(data_y), dtype='int')
    shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

train_set_x, train_set_y = shared_dataset(mnist.read(range(10),'training',path))
test_set_x, test_set_y = shared_dataset(mnist.read(range(10),'testing',path))

nTrain          = train_set_x.get_value(borrow=True).shape[0] # Number training samples
nTest           = test_set_x.get_value(borrow=True).shape[0]  # Number of test samples
batch_size      = nTrain                                      # Size of minibatches
test_batch_size = nTest                                       # Size of test batches
nTrainBatches   = nTrain / batch_size                         # Number of minibatches
nTestBatches    = nTest / test_batch_size          

#================== Load the Weights ==========================#
# theta = scipy.io.loadmat(weightFile)['opttheta'].flatten()
# cW1 = theano.shared(theta[:visibleSize*hiddenSize].reshape(visibleSize,hiddenSize))
# cW2 = theano.shared(theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(hiddenSize,visibleSize))
# cb1 = theano.shared(theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize])
# cb2 = theano.shared(theta[2*hiddenSize*visibleSize+hiddenSize:])

#================== Compute Cost ==========================#
index = T.lscalar()      # Index into the batch of training examples
x = T.matrix('x')        # Training data
y = T.ivector('y')       # Training labels
theta = theano.shared(value=np.zeros(nParams,dtype=theano.config.floatX),name='theta',borrow=True)
W = theta[:inputSize*nClasses].reshape((inputSize, nClasses))
b = theta[inputSize*nClasses:]
p_y_given_x = T.nnet.softmax(T.dot(x, W)+b)
cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
y_pred = T.argmax(p_y_given_x, axis=1)
accuracy = T.mean(T.neq(y_pred, y))

#================== Theano Functions ==========================#
batch_cost = theano.function(
    inputs=[index],
    outputs=cost,
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size],
            y:train_set_y[index * batch_size: (index + 1) * batch_size]},
    name="batch_cost")

test_cost = theano.function(
    inputs=[index],
    outputs=cost,
    givens={x:test_set_x[index * test_batch_size: (index + 1) * test_batch_size],
            y:test_set_y[index * test_batch_size: (index + 1) * test_batch_size]},
    name="batch_cost")

batch_grad = theano.function(
    inputs=[index],
    outputs=T.grad(cost, theta),
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size],
            y:train_set_y[index * batch_size: (index + 1) * batch_size]},
    name="batch_grad")

test_model = theano.function(
    inputs=[index],
    outputs=accuracy,
    givens={x:test_set_x[index * test_batch_size: (index + 1) * test_batch_size],
            y:test_set_y[index * test_batch_size: (index + 1) * test_batch_size]},
    name="test_model")

def trainFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i) for i in xrange(nTrainBatches)]
    meanLoss = np.mean(train_losses)
    return meanLoss

def gradFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    grad = np.array(batch_grad(0))
    for i in xrange(1, nTrainBatches):
        grad += np.array(batch_grad(i))
    return grad / nTrainBatches

def callbackFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i) for i in xrange(nTrainBatches)]
    test_losses = [test_cost(i) for i in xrange(nTestBatches)]
    test_acc = [test_model(i) for i in xrange(nTestBatches)]
    print 'Epoch %d Train: %f Test: %f Accuracy: %f'%(callbackFn.epoch,np.mean(train_losses),np.mean(test_losses),1-np.mean(test_acc))
    sys.stdout.flush()
    callbackFn.epoch += 1
callbackFn.epoch = 0

#================== Optimize ==========================#
x0 = (0.005*rng.randn(nParams)).astype('float32')
start = time.time()
opttheta = scipy.optimize.fmin_cg(
    f=trainFn,
    x0=x0,
    fprime=gradFn,
    callback=callbackFn,
    maxiter=nEpochs)
end = time.time()
print 'Elapsed Time(s): ', end - start
