import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import numpy.linalg
rng = np.random

def dct_matrix(rows, cols, unitary=True):
    """
    Return a (rows x cols) matrix implementing a discrete cosine transform.

    This algorithm is adapted from Dan Ellis' Rastmat
    spec2cep.m, lines 15 - 20.
    """
    rval = np.zeros((rows, cols))
    col_range = np.arange(cols)
    scale = np.sqrt(2.0/cols)
    for i in xrange(rows):
        rval[i] = np.cos(i * (col_range*2+1)/(2.0 * cols) * np.pi) * scale

    if unitary:
        rval[0] *= np.sqrt(0.5)
    return rval


class dct:
    """
    DCT class is capable of doing 1 and 2 dimensional dct transforms.
    Inputs: currShape: Tuple containing the shape of the input
            targetShape: Tuple containing the shape of the resized input (optional)
    Returns: 1/2 dimensional dct & idct transforms
    """

    def __init__(self, currShape, targetShape=None):
        """
        Pre-generate the matrices used to do the dct transform
        """
        self.transforms = []
        self.inverses = []
        self.shared_inverses = []

        for i in range(len(currShape)):
            if targetShape is None:
                t = dct_matrix(currShape[i],currShape[i])
                t_inv = numpy.linalg.inv(t)
            else:
                t = dct_matrix(targetShape[i], targetShape[i])
                t_inv = numpy.linalg.inv(t)
                # Reshape these 
                t = t[:,:currShape[i]]
                t_inv = t_inv[:,:currShape[i]]

            self.transforms.append(t)
            self.inverses.append(t_inv)
            t_inv_shared = theano.shared(value=self.inverses[i].astype('float32'),
                                         name='t_inv_shared',borrow=True)
            self.shared_inverses.append(t_inv_shared)

    def dct(self, x):
        return np.dot(self.transforms[0], x)

    def dct2(self, x):
        return np.dot(self.transforms[1], np.dot(self.transforms[0], x).T).T

    def idct(self, x):
        return T.dot(self.shared_inverses[0], x)

    def idct2(self, x):
        return T.dot(self.shared_inverses[1], T.dot(self.shared_inverses[0], x).T).T

