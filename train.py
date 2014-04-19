import sys
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mnist
import scipy.optimize
import argparse
from itertools import *
# from autoencoder import *
# from softmax import *
from model import *
from layer import *

parser = argparse.ArgumentParser(description='Testing dct transforms')
parser.add_argument('--model', required=True, type=str, help='Model training schedule')
parser.add_argument('--autoencoder', required=False, type=str, default='autoencoder')
parser.add_argument('--compression', required=False, type=float, default=.5)
parser.add_argument('--nEpochs', required=False, type=int, default=200)
parser.add_argument('--outputPrefix', required=False, type=str, default='out')
parser.add_argument('--path', required=False, default='.')
parser.add_argument('--dataDCT', action='store_true', default=False)
parser.add_argument('--beta', required=False, type=float, default=3)
parser.add_argument('--spar', required=False, type=float, default=.1)
parser.add_argument('--Lambda', required=False, type=float, default=3e-3)
parser.add_argument('--save', required=False, type=str)
parser.add_argument('--load', required=False, type=str)
args = parser.parse_args()

#=================== Parameters ===========================#
inputShape    = (28, 28)             # Dimensionality of input 
visibleSize   = 28*28                # Number of input units 
hiddenSize    = 14*14                # Number of hidden units 
sched         = args.model           # Training schedule file
Lambda        = args.Lambda          # Weight decay term
beta          = args.beta            # Weight of sparsity penalty term       
spar          = args.spar            # Sparsity parameter
compression   = args.compression     # Percentage compression
path          = args.path            # Directory to load/save files
outputPrefix  = args.outputPrefix    # Prefix for output file names
trainEpochs   = args.nEpochs         # How many epochs to train
dataDCT       = args.dataDCT         # Performs DCT transform on the dataset
aeType        = args.autoencoder     # What type of autoencoder
nStripes      = 9                    # Number of stripes for the stripeAE

#================== Load the dataset ==========================#
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    data_y = np.fromiter(chain.from_iterable(data_y), dtype='int')
    shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

datapath = path + '/data/'
train_set_x, train_set_y = shared_dataset(mnist.read(range(10),'training',datapath))
test_set_x, test_set_y = shared_dataset(mnist.read(range(10),'testing',datapath))
nTrain        = train_set_x.shape[0].eval() # Number training samples
nTest         = test_set_x.shape[0].eval()  # Number of test samples
batch_size    = 1000                      # Size of minibatches
nTrainBatches = 1#max(1,nTrain/batch_size)
nTestBatches  = 1#max(1,nTest/batch_size)

index = T.lscalar()                     # Index into the batch of training examples
x = T.matrix('x')                       # Training data 
y = T.ivector('y')                      # Vector of labels

epoch = 0

def train(nEpochs=trainEpochs):
    #================== Theano Functions ==========================#
    batch_cost = theano.function( # Compute the cost of a minibatch
        inputs=[index],
        outputs=cost,
        givens={x:train_set_x[index * batch_size: (index + 1) * batch_size],
                y:train_set_y[index * batch_size: (index + 1) * batch_size]},
        on_unused_input='ignore', name="batch_cost")

    test_cost = theano.function( # Compute the cost of a minibatch
        inputs=[index],
        outputs=cost,
        givens={x:test_set_x[index * batch_size: (index + 1) * batch_size],
                y:test_set_y[index * batch_size: (index + 1) * batch_size]},
        on_unused_input='ignore', name="test_cost")

    batch_grad = theano.function( # Compute the gradient of a minibatch
        inputs=[index],
        outputs=T.grad(cost, model.getTheta()),
        givens={x:train_set_x[index * batch_size: (index + 1) * batch_size],
                y:train_set_y[index * batch_size: (index + 1) * batch_size]},
        on_unused_input='ignore', name="batch_grad")

    if model.hasClassifier:
        test_model = theano.function(
            inputs=[index],
            outputs=accuracy,
            givens={x:test_set_x[index * batch_size: (index + 1) * batch_size],
                    y:test_set_y[index * batch_size: (index + 1) * batch_size]},
            name="test_model")

    def trainFn(theta_value):
        model.setTheta(theta_value)
        loss = batch_cost(0)
        for i in xrange(1, nTrainBatches):
            loss += batch_cost(i)
        return loss / nTrainBatches

    def gradFn(theta_value):
        model.setTheta(theta_value)
        grad = batch_grad(0)
        for i in xrange(1, nTrainBatches):
            grad += batch_grad(i)
        try:
            return grad / nTrainBatches
        except:
            if type(grad) == theano.sandbox.cuda.CudaNdarray:
                return np.array(grad.__array__()) / nTrainBatches

    def callbackFn(theta_value):
        global epoch
        model.setTheta(theta_value)
        train_losses = [batch_cost(i) for i in xrange(nTrainBatches)]
        test_losses = [test_cost(i) for i in xrange(nTestBatches)]
        print('Epoch %d Train: %f Test: %f'%(epoch,np.mean(train_losses),np.mean(test_losses))),
        if model.hasClassifier:
            test_acc = [test_model(i) for i in xrange(nTestBatches)]
            print 'Accuracy: %.2f'%(100*(1-np.mean(test_acc)))
        else: print ''
        sys.stdout.flush()
        epoch += 1

    #================== Optimize ==========================#
    start = time.time()
    opttheta = scipy.optimize.fmin_cg(
        f=trainFn,
        x0=model.getTheta().get_value(),
        fprime=gradFn,
        callback=callbackFn,
        maxiter=nEpochs)
    end = time.time()
    print 'Elapsed Time(s): ', end - start
    return opttheta

# Run the training schedule 
execfile(sched)
# f = open(sched,'rb')
# for line in f:
#     exec(line)
