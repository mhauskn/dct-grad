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
    X = T.dot(T.cos((a + .5) * (np.pi/N * a.T)), x)
    return X

# iDCT - Inverse 1-dimensional dct. 
# Should be applied to a vector of dct coefficients.
# Returns a vector of the same dimension containing the reconstructed elements.
def idct(x):
    b = T.matrix('b')
    N = x.shape[0]
    a = T.ones([N,1]) * T.arange(N)
    b = T.cos((a.T + .5) * (np.pi/N * a))
    c = T.set_subtensor(b[:,0], .5)
    X = T.dot(c, x) * (2./N)
    return X

# dct2 - 2-dimensional dct transform
# Should be applied to a matrix x.
# Returns a matrix of dct coefficients with the same dimension as x.
def dct2(x):
    ''' Apply 1D dct to the rows & columns of x '''
    result, updates = theano.scan(dct,sequences=[x])
    result2, updates2 = theano.scan(dct, sequences=[result.T])
    return result2.T

# idct2 - Inverse 2-dimensional dct.
# Should be applied to a matrix of dct coefficients x.
# Returns a matrix of the same dimension containing the reconstructed elements.
def idct2(x):
    ''' Apply 1D idct to the rows & columns of x '''
    result, updates = theano.scan(idct,sequences=[x])
    result2, updates2 = theano.scan(idct, sequences=[result.T])
    return result2.T

r = T.iscalar('r')
c = T.iscalar('c')


def idct2Expand(x, r, c):
    ''' Apply 1D idct to the rows & columns of x '''
    result, updates = theano.scan(idctExpand,sequences=[x],non_sequences=[r])
    result2, updates2 = theano.scan(idctExpand, sequences=[result.T],non_sequences=[c])
    return result2.T

def idctExpand(x, N):
    b = T.matrix('b')
    a = T.ones([N,1]) * T.arange(N)
    b = T.cos((a.T + .5) * (np.pi/N * a))
    c = T.set_subtensor(b[:,0], .5)
    z = T.zeros([N],dtype='float32')
    X = T.dot(c, T.set_subtensor(z[:x.size], x)) * (2./N)
    return X

# X = idct2Expand(x,r,c)
# dc = theano.function(inputs=[x,r,c], outputs=X, allow_input_downcast=True, on_unused_input='ignore')
# v = np.ones((2,2))
# print v
# print dc(v,5,5)
# X = idctExpand(x,N)
# dc = theano.function(inputs=[x,N], outputs=X, allow_input_downcast=True, on_unused_input='ignore')
# v = rng.randn(5)
# print v
# print dc(v,20)

    
