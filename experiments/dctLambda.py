# This experiment is attempting to find the relationship between dct weight penalties and normal weight penalty
import cluster

j, expDir = cluster.getEnv()

dctLambda = 3e-4
while dctLambda < 3:
    j.setArgs('%s/reshapingAE.py --noWeightCost --path %s --outputPrefix reshapingAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    j.setOutputPrefix('%s/results/reshapingAE%f'%(expDir,dctLambda))
    j.submit()

    j.setArgs('%s/stripeAE.py --noWeightCost --path %s --outputPrefix stripeAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    j.setOutputPrefix('%s/results/stripeAE%f'%(expDir,dctLambda))
    j.submit()

    j.setArgs('%s/rectangleAE.py --noWeightCost --path %s --outputPrefix rectangleAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    j.setOutputPrefix('%s/results/rectangleAE%f'%(expDir,dctLambda))
    j.submit()

    dctLambda *= 10
