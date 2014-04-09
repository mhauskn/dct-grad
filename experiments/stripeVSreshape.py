# This experiment is meant to test the power of dct encoding using a rectangle of coeffs v. stripes v. reshaping
import condor, socket, subprocess

local = socket.gethostname() == 'drogba'
nStripes = 10
execStr = 'python' if local else '/lusr/bin/python'
expDir = '/home/matthew/projects/dct-grad/' if local else '/u/mhauskn/projects/dct-grad/'
cj = condor.job(execStr)
for stripe in range(1,nStripes):
    cj.setArgs('%s/stripeAE.py --nStripes %d --path %s --outputPrefix stripe%d'%(expDir,stripe,expDir, stripe))
    cj.setOutputPrefix('%s/results/stripe%d'%(expDir,stripe))
    cj.runLocal() if local else cj.submit()

    nStripeParams = int(28 * (stripe - .5)) * 196 * 2 + 196 + 784
    perc = nStripeParams/float(308308) # divide by total params

    cj.setArgs('%s/reshapingAE.py --compression %f --path %s --outputPrefix reshape%d'%(expDir,perc,expDir,stripe))
    cj.setOutputPrefix('%s/results/reshape%d'%(expDir,stripe))
    cj.runLocal() if local else cj.submit()
    
    cj.setArgs('%s/rectangleAE.py --compression %f --path %s --outputPrefix rectangle%d'%(expDir,perc,expDir,stripe))
    cj.setOutputPrefix('%s/results/rectangle%d'%(expDir,stripe))
    cj.runLocal() if local else cj.submit()
