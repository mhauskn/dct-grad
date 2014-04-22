import numpy as np
import theano
import theano.tensor as T
import pickle

class Model():
    def __init__(self):
        self.layers = []
        self.params = []
        self.frozen = []
        self.theta = theano.shared(value=np.zeros(0,dtype=theano.config.floatX), name='theta', borrow=True)

    def hasClassifier(self):
        l = self.layers[-1]
        return hasattr(l,'accuracy') and callable(getattr(l,'accuracy'))

    def addLayer(self,layer):
        self.layers.append(layer)
        self.params.append(layer.getNParams())
        self.frozen.append(False)
        lx0 = layer.getx0()
        newtheta = np.concatenate([self.theta.get_value(), lx0]).astype('float32')
        self.theta.set_value(newtheta)

    def deleteLayer(self):
        assert len(self.layers) > 0
        l = self.layers.pop()
        p = self.params.pop()
        self.theta.set_value(self.theta.get_value()[:-p])

    def freezeLayer(self, indx):
        assert indx >= 0 and indx < len(self.layers) and not self.frozen[indx]
        n = 0
        for i, l in enumerate(self.layers):
            p = self.params[i]
            if i == indx:
                t = self.theta.get_value()
                assert p == l.getNParams()
                assert len(t) >= n+p
                l.setTheta(t[n:n+p])
                self.theta.set_value(np.concatenate([t[:n],t[n+p:]]).astype('float32'))
                break
            if not self.frozen[i]:
                n += p
        self.frozen[indx] = True

    def unfreezeLayer(self, indx):
        assert indx >= 0 and indx < len(self.layers) and self.frozen[indx]
        n = 0
        for i, l in enumerate(self.layers):
            p = self.params[i]
            if i == indx:
                t = self.theta.get_value()
                layerParams = l.getTheta()
                assert len(layerParams) == p
                self.theta.set_value(np.concatenate([t[:n],layerParams,t[n:]]).astype('float32'))
            n += p
        self.frozen[indx] = False

    def finalize(self):
        n = 0
        for l,p,frozen in zip(self.layers,self.params,self.frozen):
            if not frozen:
                l.setTheta(self.theta[n:n+p])
                n += p
        print self

    def getTheta(self):
        return self.theta

    def getOutputSize(self):
        return self.layers[-1].getOutputSize()

    def getWeights(self):
        return [l.getWeights() for l in self.layers]

    def setTheta(self, theta_value):
        self.theta.set_value(theta_value, borrow=True)
        n = 0
        for l,p,frozen in zip(self.layers,self.params,self.frozen):
            if not frozen:
                l.setTheta(self.theta[n:n+p])
                n += p

    def getx0(self):
        return np.concatenate([l.getx0() for l in self.layers]).astype('float32')

    def forward(self, x):
        a = x
        for l in self.layers:
            a = l.forward(a)
        return a

    def save(self, fname='model.pkl'):
        pickle.dump(self.__dict__,open(fname,'wb'))

    def load(self, fname='model.pkl'):
        tmp = pickle.load(open(fname,'rb'))
        self.__dict__.update(tmp)

    def saveImages(self, path, opttheta):
        self.setTheta(opttheta)
        l = self.layers[0]
        fname = path + '.png'
        if hasattr(l,'saveImage') and callable(getattr(l,'saveImage')):
            l.saveImage(fname)
                    
    def __str__(self):
        s = ''
        for i, l in enumerate(self.layers):
            if self.frozen[i]:
                s += 'Layer %d [Frozen]: '%(i) + l.__str__() + '\n'
            else:
                s += 'Layer %d: '%(i) + l.__str__() + '\n'
        s += 'Theta Size: %d'%len(self.theta.get_value())
        return s
        
