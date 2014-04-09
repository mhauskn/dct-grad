# This experiment is attempting to find a good value of lambda for the Penalize-if-nonzero regularization
import cluster

j, expDir = cluster.getEnv()

dctLambda = 3e-6
while dctLambda < 3e3:
    j.setArgs('%s/reshapingAE.py --noWeightCost --path %s --outputPrefix reshapingAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    j.setOutputPrefix('%s/results/reshapingAE%f'%(expDir,dctLambda))
    j.submit()

    dctLambda *= 10
