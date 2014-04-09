import sys, time, theano, dct, mnist, scipy.optimize, argparse, PIL.Image, numpy as np, theano.tensor as T, matplotlib.pyplot as plt, matplotlib.cm as cm, utils, pickle

rng = np.random

parser = argparse.ArgumentParser(description='Testing dct transforms')
parser.add_argument('--compression', required=False, type=float, default=.5)
parser.add_argument('--nEpochs', required=False, type=int, default=200)
parser.add_argument('--outputPrefix', required=False, type=str, default='out')
parser.add_argument('--path', required=False, default='.')
parser.add_argument('--load', required=False, type=str)
parser.add_argument('--save', required=False, type=str)
parser.add_argument('--noKLDiv', action='store_true', default=False)
parser.add_argument('--noWeightCost', action='store_true', default=False)
parser.add_argument('--noDCTWeightCost', action='store_true', default=False)
parser.add_argument('--dataDCT', action='store_true', default=False)
parser.add_argument('--dctLambda', required=False, type=float, default=3e-3)
args = parser.parse_args()

#=================== Parameters ===========================#
visibleSize  = 28*28             # Number of input units 
hiddenSize   = 14*14             # Number of hidden units 
alpha        = 9e-1              # Learning rate
Lambda       = 3e-3              # Weight decay term
dctLambda    = args.dctLambda    # DCT-weight decay term
beta         = 3                 # Weight of sparsity penalty term       
spar         = 0.1               # Sparsity parameter
path         = args.path         # Directory to load/save files
outputPrefix = args.outputPrefix # Prefix for output file names
nStripes     = args.nStripes     # Number of bands of coeffs to learn over
stripeWidth  = args.stripeWidth  # How wide each stripe is
trainEpochs  = args.nEpochs      # How many epochs to train
dataDCT      = args.dataDCT      # Performs DCT transform on the dataset
useDCT       = nStripes > 0      # Enable dct compression

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
if useDCT:
    dctW1Shape = (int(np.round(np.sqrt(compression*visibleSize*visibleSize))),
                  int(np.round(np.sqrt(compression*hiddenSize*hiddenSize))))
    dctW2Shape = (dctW1Shape[1],dctW1Shape[0])
    dctWeightSize = dctW1Shape[0]*dctW1Shape[1]
    nDCTParams = 2*dctWeightSize + nBiasParams

    theta = theano.shared(value=np.zeros(nDCTParams,dtype=theano.config.floatX),
                          name='theta',borrow=True)

    print "Learning in DCT space\n%d total parameters (%f%%)\nDCT input shape"\
        %(nDCTParams,100.*nDCTParams/nParams), dctW1Shape

    cW1 = theta[:dctWeightSize].reshape(dctW1Shape)
    cW2 = theta[dctWeightSize:2*dctWeightSize].reshape(dctW2Shape)
    b1 = theta[2*dctWeightSize:2*dctWeightSize+hiddenSize]
    b2 = theta[2*dctWeightSize+hiddenSize:]

    d1 = dct.dct(dctW1Shape, W1Shape)
    d2 = dct.dct(dctW2Shape, W2Shape)

    W1 = d1.idct2(cW1)
    W2 = d2.idct2(cW2)
else:
    print "Directly learning weights and biases (no DCT)\n%d total parameters." % nParams
    theta = theano.shared(value=np.zeros(nParams,dtype=theano.config.floatX),name='theta',borrow=True)
    W1 = theta[:visibleSize*hiddenSize].reshape(W1Shape)
    W2 = theta[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape(W2Shape)
    b1 = theta[2*visibleSize*hiddenSize:2*visibleSize*hiddenSize+hiddenSize]
    b2 = theta[2*visibleSize*hiddenSize+hiddenSize:]

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
    pdf = np.vectorize(scipy.stats.norm().pdf)

    wtmp = np.outer(pdf(np.linspace(0,2,nDctRows)),pdf(np.linspace(0,2,hiddenSize)))
    cWPenalty = 1.-(wtmp/np.max(wtmp))
    
    penW = theano.shared(value=cWPenalty.astype('float32'),borrow=True)    

    dctWeightDecayPenalty = (dctLambda/2.) * (T.sum(penW * T.abs_(cW1)) + T.sum(penW * T.abs_(cW2)))
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
if args.load:
    print 'Loading parameters from file', args.load
    x0 = pickle.load(open(args.load,'rb'))
else:
    r = np.sqrt(6) / np.sqrt(visibleSize+hiddenSize+1)
    x0 = np.concatenate(((rng.randn(nWeightParams)*2*r-r).flatten(),
                         np.zeros(nBiasParams))).astype('float32')
    if useDCT:
        x0w1 = x0[:visibleSize*hiddenSize].reshape(W1Shape)
        dctShrink_cW1 = dct.dct(W1Shape)
        iW1 = dctShrink_cW1.dct2(x0w1)[:dctW1Shape[0],:dctW1Shape[1]]
        x0w2 = x0[visibleSize*hiddenSize:2*visibleSize*hiddenSize].reshape(W2Shape)
        dctShrink_cW2 = dct.dct(W2Shape)
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

#================== Save Weights ==========================#
if args.save:
    pickle.dump(opttheta, open(args.save, 'wb'))

#================== Save W1 Image ==========================#
fname = path + '/results/' + outputPrefix + '.png'
theta.set_value(opttheta, borrow=True)
image = PIL.Image.fromarray(utils.tile_raster_images(
        X=W1.eval().T,
        img_shape=(28, 28), tile_shape=(14, 14),
        tile_spacing=(1, 1)))
image.save(fname)
