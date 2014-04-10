# This experiment is attempting to find a good value of lambda for the Penalize-if-nonzero regularization
import cluster

j, expDir = cluster.getEnv()

Lambda = 3e-6
while Lambda < 3e3:
    j.setArgs('%s/train.py --autoencoder autoencoder --path %s --outputPrefix ae%f --Lambda %f'%(expDir,expDir,Lambda,Lambda))
    j.setOutputPrefix('%s/results/reshapingAE%f'%(expDir,Lambda))
    j.submit()

    Lambda *= 10
