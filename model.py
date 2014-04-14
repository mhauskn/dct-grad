import numpy as np
import theano
import theano.tensor as T
import pickle

class Model():
    def __init__(self):
        self.layers = []
        self.params = []
        self.hasClassifier = False

    def addLayer(self,layer):
        self.layers.append(layer)
        self.params.append(layer.getNParams())
        self.hasClassifier = hasattr(layer,'accuracy') and callable(getattr(layer,'accuracy'))

    def finalize(self):
        totalParams = int(np.sum(self.params))
        if not hasattr(self, 'theta'): 
            self.theta = theano.shared(value=np.zeros(totalParams,dtype=theano.config.floatX),
                                       name='theta',borrow=True)
            self.setTheta(self.getx0())

    def setTheta(self, theta_value):
        self.theta.set_value(theta_value, borrow=True)
        n = 0
        for l,p in zip(self.layers,self.params):
            l.setTheta(self.theta[n:n+p])
            n += p

    def getx0(self):
        return np.concatenate([l.getx0() for l in self.layers]).astype('float32')

    def forward(self, x):
        a = x
        for l in self.layers:
            a = l.forward(a)
        return a

    def cost(self, x, output, labels):
        return self.layers[-1].cost(x, output, labels)

    def accuracy(self, output, labels):
        return self.layers[-1].accuracy(output, labels)

    def save(self, fname='model.pkl'):
        pickle.dump(self.__dict__,open(fname,'wb'))

    def load(self, fname='model.pkl'):
        tmp = pickle.load(open(fname,'rb'))
        self.__dict__.update(tmp)

    def saveImages(self, path, opttheta):
        self.setTheta(opttheta)
        for i in xrange(len(self.layers)):
            l = self.layers[i]
            fname = path + 'L' + str(i) + '.png'
            if hasattr(l,'saveImage') and callable(getattr(l,'saveImage')):
                l.saveImage(fname)
                    
    def __str__(self):
        for i in xrange(len(self.layers)):
            print 'Layer %d:'%(i), self.layers[i]
        
