# This experiment is meant to test the power of dct encoding using a rectangle of coeffs v. stripes v. reshaping

import condor, socket, subprocess

nStripes = 10
cj = condor.job('python')
for stripe in range(1,nStripes):
    cj.setArgs('/home/matthew/projects/dct-grad/autoencoder.py --nStripes %d --path /home/matthew/projects/dct-grad/ --outputPrefix stripe%d'%stripe)
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/stripe%d'%stripe)
    cj.runLocal()

    nStripeParams = int(28 * (stripe - .5)) * 196 * 2 + 196 + 784
    perc = nStripeParams/float(308308) # divide by total params

    cj.setArgs('/home/matthew/projects/dct-grad/reshapingAE.py --compression %f --path /home/matthew/projects/dct-grad/ --outputPrefix reshape%d'%stripe)
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/reshape%d'%stripe)
    cj.runLocal()
    
