# This experiment is attempting to find the relationship between dct weight penalties and normal weight penalty
import condor, socket, subprocess

local = socket.gethostname() == 'drogba'
execStr = 'python' if local else '/lusr/bin/python'
expDir = '/home/matthew/projects/dct-grad/' if local else '/u/mhauskn/projects/dct-grad/'
cj = condor.job(execStr)
dctLambda = 3e-4
while dctLambda < 3e2:
    cj.setArgs('%s/reshapingAE.py --path %s --outputPrefix dctLambda%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    cj.setOutputPrefix('%s/results/dctLambda%f'%(expDir,stripe,dctLambda))
    cj.runLocal() if local else cj.submit()

    cj.setArgs('%s/reshapingAE.py --noWeightCost --path %s --outputPrefix dctLambdaNoWTCost%f --dctLambda %f'%(expDir,expDir,dctLambda,dctLambda))
    cj.setOutputPrefix('%s/results/dctLambdaNoWtCost%f'%(expDir,stripe,dctLambda))
    cj.runLocal() if local else cj.submit()

    dctLambda *= 10
