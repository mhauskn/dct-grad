import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import dct
rng = np.random

visibleSize = 28*28 # number of input units 
hiddenSize = 196    # number of hidden units 
sparsityParam = 0.1 # desired average activation of the hidden units.
Lambda = 100        # weight decay parameter       
beta = 3            # weight of sparsity penalty term       

# Load the dataset
import cPickle, gzip
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)


m = test_set[0].shape[0] # Number training samples
batch_size = m
n_train_batches = m / batch_size
training_epochs = 5

index = T.lscalar() # Index into the batch of training examples
x = T.matrix('x')   # Training data

# Initialize weights
r  = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)

# DCT Coefficients matrices for weights and biases
cW1 = theano.shared(rng.randn(visibleSize, hiddenSize) * 2 * r - r, name='C1')
cW2 = theano.shared(rng.randn(hiddenSize, visibleSize) * 2 * r - r, name='C2')
cb1 = theano.shared(np.zeros(hiddenSize))
cb2 = theano.shared(np.zeros(visibleSize))

# Use the inverse DCT transform to recover the weights/biases of the network
W1 = dct.idct2(cW1)
W2 = dct.idct2(cW2)
b1 = dct.idct(cb1)
b2 = dct.idct(cb2)

# Forward Propagate
a1 = T.nnet.sigmoid(T.dot(x, W1) + b1)
a2 = T.nnet.sigmoid(T.dot(a1, W2) + b2)

# Cost is the reconstruction error only
cost = T.sum((a2 - x) ** 2) / (2 * m)

# Gradient of cost wrt dct coefficients
gw1, gb1, gw2, gb2 = T.grad(cost, [cW1, cb1, cW2, cb2]) 

train = theano.function(
          inputs=[index],
          outputs=[cost],
          updates=((cW1, cW1 - Lambda * gw1), (cb1, cb1 - Lambda * gb1),
                   (cW2, cW2 - Lambda * gw2), (cb2, cb2 - Lambda * gb2)),
          givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]})
predict = theano.function(inputs=[x], outputs=[a1,a2,cost])

start = time.time()
for epoch in xrange(training_epochs):
    for batch_index in xrange(n_train_batches):
        print train(batch_index)[0]
end = time.time()
print 'Elapsed Time (seconds): ', end - start


