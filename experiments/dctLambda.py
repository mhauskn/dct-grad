# This experiment is attempting to find the relationship between dct weight penalties and normal weight penalty
import condor, socket, subprocess

local = socket.gethostname() == 'drogba'
execStr = 'python' if local else '/lusr/bin/python'
expDir = '/home/matthew/projects/dct-grad/' if local else '/u/mhauskn/projects/dct-grad/'
cj = condor.job(execStr)
dctLambda = 3e-4
while dctLambda < 3:
    cj.setArgs('%s/reshapingAE.py --noWeightCost --path %s --outputPrefix reshapingAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    cj.setOutputPrefix('%s/results/reshapingAE%f'%(expDir,dctLambda))
    cj.runLocal() if local else cj.submit()

    cj.setArgs('%s/stripeAE.py --noWeightCost --path %s --outputPrefix stripeAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    cj.setOutputPrefix('%s/results/stripeAE%f'%(expDir,dctLambda))
    cj.runLocal() if local else cj.submit()

    cj.setArgs('%s/rectangleAE.py --noWeightCost --path %s --outputPrefix rectangleAE%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    cj.setOutputPrefix('%s/results/rectangleAE%f'%(expDir,dctLambda))
    cj.runLocal() if local else cj.submit()

    dctLambda *= 2
