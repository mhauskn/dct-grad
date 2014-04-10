# This experiment is attempting to find a good value of lambda for the Penalize-if-nonzero regularization
import cluster

j, expDir = cluster.getEnv()

Lambda = 3e-4
while Lambda < 3e3:
    pre = 'ae' + str(Lambda)
    j.setArgs('%s/train.py --autoencoder autoencoder --path %s --outputPrefix %s --Lambda %f'%(expDir,expDir,pre,Lambda))
    j.setOutputPrefix('%s/results/%s'%(expDir,pre))
    j.submit()

    Lambda *= 10
