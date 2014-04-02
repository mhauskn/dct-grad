# This experiment is meant to test the power of dct encoding using a rectangle of coeffs v. stripes v. reshaping
import condor, socket, subprocess

nStripes = 10
cj = condor.job('python')
for stripe in range(1,nStripes):
    cj.setArgs('/home/matthew/projects/dct-grad/stripeAE.py --nStripes %d --path /home/matthew/projects/dct-grad/ --outputPrefix stripe%d'%(stripe,stripe))
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/stripe%d'%stripe)
    cj.runLocal() if socket.gethostname() == 'drogba' else cj.submit()

    nStripeParams = int(28 * (stripe - .5)) * 196 * 2 + 196 + 784
    perc = nStripeParams/float(308308) # divide by total params

    cj.setArgs('/home/matthew/projects/dct-grad/reshapingAE.py --compression %f --path /home/matthew/projects/dct-grad/ --outputPrefix reshape%d'%(perc,stripe))
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/reshape%d'%stripe)
    cj.runLocal() if socket.gethostname() == 'drogba' else cj.submit()
    
    cj.setArgs('/home/matthew/projects/dct-grad/rectangleAE.py --compression %f --path /home/matthew/projects/dct-grad/ --outputPrefix rectangle%d'%(perc,stripe))
    cj.setOutputPrefix('/home/matthew/projects/dct-grad/results/rectangle%d'%stripe)
    cj.runLocal() if socket.gethostname() == 'drogba' else cj.submit()
