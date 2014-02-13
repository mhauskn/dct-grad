import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import dct
rng = np.random

visibleSize = 28*28 # number of input units 
hiddenSize = 196    # number of hidden units 
alpha = 100         # learning rate
ds = 10             # downscaling factor

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
cW1 = theano.shared(rng.randn(visibleSize/ds, hiddenSize/ds) * 2 * r - r, name='C1')
cW2 = theano.shared(rng.randn(hiddenSize/ds, visibleSize/ds) * 2 * r - r, name='C2')
cb1 = theano.shared(np.zeros(hiddenSize/ds))
cb2 = theano.shared(np.zeros(visibleSize/ds))

# Expand the coefficients into larger matrices
W1 = T.zeros((visibleSize, hiddenSize))
W1 = dct.idct2(T.set_subtensor(W1[:visibleSize/ds,:hiddenSize/ds], cW1))
W2 = T.zeros((hiddenSize, visibleSize))
W2 = dct.idct2(T.set_subtensor(W2[:hiddenSize/ds,:visibleSize/ds], cW2))
b1 = T.zeros([hiddenSize])
b1 = dct.idct(T.set_subtensor(b1[:hiddenSize/ds], cb1))
b2 = T.zeros([visibleSize])
b2 = dct.idct(T.set_subtensor(b2[:visibleSize/ds], cb2))

# Use the inverse DCT transform to recover the weights/biases of the network
# W1 = dct.idct2(cW1)
# W2 = dct.idct2(cW2)
# b1 = dct.idct(cb1)
# b2 = dct.idct(cb2)

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
          updates=((cW1, cW1 - alpha * gw1), (cb1, cb1 - alpha * gb1),
                   (cW2, cW2 - alpha * gw2), (cb2, cb2 - alpha * gb2)),
          givens={x:train_set_x[index * batch_size: (index + 1) * batch_size]})
# theano.printing.pydotprint(train,'graph.png')

predict = theano.function(inputs=[x], outputs=[a1,a2,cost])

start = time.time()
for epoch in xrange(training_epochs):
    for batch_index in xrange(n_train_batches):
        print train(batch_index)[0]
end = time.time()
print 'Elapsed Time (seconds): ', end - start


