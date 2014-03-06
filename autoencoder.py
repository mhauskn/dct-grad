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
parser.add_argument('--process_num', metavar='processNum', required=False,
                    type=int, help='Condor process number')
parser.add_argument('--path', metavar='path', required=False, default='.',
                    help='Condor process number')
args = parser.parse_args()

visibleSize   = 28*28 # number of input units 
hiddenSize    = 14*14 # number of hidden units 
alpha         = 9e-1  # learning rate
Lambda        = 3e-3  # weight decay term
beta          = 3     # weight of sparsity penalty term       
spar          = 0.1   # sparsity parameter
useDCT        = True  # enable dct compression
v             = visibleSize # Effective visual size
h             = hiddenSize  # Effective hidden size
path          = args.path

if args.process_num is not None:
    assert int(args.process_num) > 0 and int(args.process_num) <= min(visibleSize,hiddenSize)
    v = int(args.process_num) # Effective visual size
    h = int(args.process_num)  # Effective hidden size
    print 'Using dct coeff matrix of size [%d, %d]' % (v,h)

# Load the dataset
images, labels = mnist.read(range(10),'training',path)
images = images / 255. # Remap between [0,1]
patches = images[:10000] # Matlab patches use an odd col-major order
for i in range(len(patches)):
    patches[i,:] = patches[i,:].reshape(28,28).T.flatten()

# import cPickle, gzip
# f = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

# test_set_x, test_set_y   = shared_dataset(test_set)
# valid_set_x, valid_set_y = shared_dataset(valid_set)
# train_set_x, train_set_y = shared_dataset(train_set)
train_set_x = theano.shared(np.asarray(patches, dtype=theano.config.floatX))

m               = train_set_x.shape[0].eval() # Number training samples
batch_size      = m
n_train_batches = m / batch_size

index = T.lscalar() # Index into the batch of training examples
x = T.matrix('x')   # Training data 

# Initialize weights & biases
r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)

theta = theano.shared(value=np.zeros(2*v*h+v+h,dtype=theano.config.floatX),name='theta',borrow=True)
cW1 = theta[:v*h].reshape((v,h))
cW2 = theta[v*h:2*v*h].reshape((h,v))
cb1 = theta[2*v*h:2*v*h+h]
cb2 = theta[2*v*h+h:]

# Load saved matlab weights
# import scipy.io
# theta = scipy.io.loadmat('weights.mat')['opttheta'].flatten()
# cW1 = theano.shared(theta[:visibleSize*hiddenSize].reshape(visibleSize,hiddenSize))
# cW2 = theano.shared(theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(hiddenSize,visibleSize))
# cb1 = theano.shared(theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize])
# cb2 = theano.shared(theta[2*hiddenSize*visibleSize+hiddenSize:])

if useDCT:
    # Create the DCT objects: currShape, targetShape
    dct_cW1 = dct.dct(cW1.shape.eval(), (visibleSize, hiddenSize))
    dct_cW2 = dct.dct(cW2.shape.eval(), (hiddenSize, visibleSize))
    dct_cb1 = dct.dct(cb1.shape.eval(), (hiddenSize,))
    dct_cb2 = dct.dct(cb2.shape.eval(), (visibleSize,))

    # Expand the coefficients into larger matrices
    W1 = dct_cW1.idct2(cW1)
    W2 = dct_cW2.idct2(cW2)
    b1 = dct_cb1.idct(cb1)
    b2 = dct_cb2.idct(cb2)

else:
    W1 = cW1
    W2 = cW2
    b1 = cb1
    b2 = cb2
    
# Forward Propagate
z1 = T.dot(x, W1) + b1
a1 = T.nnet.sigmoid(T.dot(x, W1) + b1)
a2 = T.nnet.sigmoid(T.dot(a1, W2) + b2)

m = x.shape[0]
sse = T.sum((a2 - x) ** 2) / (2. * m)
avgAct = a1.mean(axis=0) # Col-Mean: AvgAct of each hidden unit across all m-examples
KL_Div = beta * T.sum(spar * T.log(spar/avgAct) + (1-spar) * T.log((1-spar)/(1-avgAct)))
weightDecayPenalty = (Lambda/2.) * (T.sum(cW1**2) + T.sum(cW2**2))
cost = sse + KL_Div + weightDecayPenalty

# Gradient of cost wrt dct coefficients
# grad = T.grad(cost, theta)
# gw1, gb1, gw2, gb2 = T.grad(cost, [cW1, cb1, cW2, cb2]) 

# Compute the cost of a minibatch
batch_cost = theano.function(
    inputs=[index],
    outputs=cost,
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_cost")

# Compute the gradient of a minibatch WRT theta
batch_grad = theano.function(
    inputs=[index],
    outputs=T.grad(cost, theta),
    givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]},
    name="batch_grad")

def trainFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in xrange(n_train_batches)]
    meanLoss = np.mean(train_losses)
    # print meanLoss
    return meanLoss

def gradFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    grad = batch_grad(0)
    for i in xrange(1, n_train_batches):
        grad += batch_grad(i * batch_size)
    return grad / n_train_batches

def callbackFn(theta_value):
    theta.set_value(theta_value, borrow=True)
    train_losses = [batch_cost(i * batch_size) for i in xrange(n_train_batches)]
    print 'Epoch',callbackFn.epoch,np.mean(train_losses)
    sys.stdout.flush()
    callbackFn.epoch += 1
callbackFn.epoch = 0

# train = theano.function(
#     inputs=[index],
#     outputs=[cost],
#     updates=(
#         #(theta, theta - alpha * grad),),
#         (cW1, cW1 - alpha * gw1), (cb1, cb1 - alpha * gb1),
#              (cW2, cW2 - alpha * gw2), (cb2, cb2 - alpha * gb2)),
#     givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]})
# predict = theano.function(inputs=[x], outputs=[cost], allow_input_downcast=True)
# print predict(patches)

training_epochs = 200
start = time.time()
opttheta = scipy.optimize.fmin_cg(
    f=trainFn,
    x0=np.concatenate(((rng.randn(2*v*h)*2*r-r).flatten(),np.zeros(v+h))).astype('float32'), # TODO: need to do this for useDCT case
    fprime=gradFn,
    callback=callbackFn,
    maxiter=training_epochs)

fname = path + '/results/' + (str(args.process_num) if args.process_num else 'final') + '.png'
theta.set_value(opttheta, borrow=True)
image = PIL.Image.fromarray(tile_raster_images(
        X=dct_cW1.idct2(cW1).eval().T if useDCT else cW1.eval().T,
        img_shape=(28, 28), tile_shape=(14, 14),
        tile_spacing=(1, 1)))
image.save(fname)

# for epoch in xrange(training_epochs):
#     for batch_index in xrange(n_train_batches):
#         print 'Epoch', epoch, train(batch_index)[0]
#         image = PIL.Image.fromarray(tile_raster_images(
#                 X=cW1.eval().T,
#                 img_shape=(28, 28), tile_shape=(14, 14),
#                 tile_spacing=(1, 1)))
#         image.save('images/epoch_%d.png'%epoch)
end = time.time()
print 'Elapsed Time(s): ', end - start


