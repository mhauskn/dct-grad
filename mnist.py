import os, struct
import dct
from array import array as pyarray
import numpy as np
from numpy import append, array, int8, uint8, zeros

def read(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in xrange(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        labels[i] = lbl[ind[i]]

    # Matlab patches use an odd col-major order
    images = images / 255. # Remap between [0,1]        
    for i in range(len(images)):  # Convert them back over to row-major
        images[i,:] = images[i,:].reshape(28,28).T.flatten()

    return images, labels

def applyDCT(images, nCoeffs=100):
    ''' Apply a 2-D DCT transform to the images '''
    edge = int(np.sqrt(nCoeffs))
    dct_images = zeros((len(images), nCoeffs), dtype='float32')
    imgDCT = dct.dct((28,28))
    for i in range(len(images)):
        a = imgDCT.dct2(images[i,:].reshape(28,28))[:edge,:edge].flatten()
        dct_images[i,:] = a #(a - min(a)) / (max(a) - min(a))
    return dct_images

def applyPCA(images, nCoeffs=100):
    ''' Apply a PCA transform to the images '''
    # images = (images.T - images.mean(1)).T # Zero-mean the data
    # a = np.dot(images.T, images) / len(images)

    import pickle
    a = pickle.load(open('eigh.pkl','r'))

    vals, vecs = np.linalg.eigh(a)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    vals = np.sqrt(vals[:nCoeffs])
    vecs = vecs[:, :nCoeffs]

    def whiten(x):
        return np.dot(x, np.dot(vecs, np.diag(1. / vals)))

    return whiten(images)


