import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
rng = np.random

x = T.vector('x')
a = T.matrix('a')
N = T.iscalar('N')

# DCT - 1-dimensional dct transform. 
# Should be applied to a vector x.
# Returns a vector of dct coefficients with the same dimension as x.
def dct(x):
    N = x.shape[0]
    a = T.ones([N,1]) * T.arange(N)
    X = T.dot(T.cos((a + .5) * (np.pi/N * T.transpose(a))), x)
    return X

# iDCT - Inverse 1-dimensional dct. 
# Should be applied to a vector of dct coefficients.
# Returns a vector of the same dimension containing the reconstructed elements.
def idct(x):
    b = T.matrix('b')
    N = x.shape[0]
    a = T.ones([N,1]) * T.arange(N)
    b = T.cos((T.transpose(a) + .5) * (np.pi/N * a))
    c = T.set_subtensor(b[:,0], .5)
    X = T.dot(c, x) * (2./N)
    return X

# dct2 - 2-dimensional dct transform
# Should be applied to a matrix x.
# Returns a matrix of dct coefficients with the same dimension as x.
def dct2(x):
    ''' Apply 1D dct to the rows & columns of x '''
    result, updates = theano.scan(dct,sequences=[x])
    result2, updates2 = theano.scan(dct, sequences=[T.transpose(result)])
    return T.transpose(result2)

# idct2 - Inverse 2-dimensional dct.
# Should be applied to a matrix of dct coefficients x.
# Returns a matrix of the same dimension containing the reconstructed elements.
def idct2(x):
    ''' Apply 1D idct to the rows & columns of x '''
    result, updates = theano.scan(idct,sequences=[x])
    result2, updates2 = theano.scan(idct, sequences=[T.transpose(result)])
    return T.transpose(result2)


