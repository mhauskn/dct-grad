import condor

nTrials = 10
nStripes = 10

cj = condor.job('/lusr/bin/python')

for trial in range(nTrials):
    for stripe in range(nStripes):
        cj.setArgs('/u/mhauskn/projects/dct-grad/autoencoder.py --nStripes %d --path /u/mhauskn/projects/dct-grad/ --outputPrefix t%d_%d'%(stripe,trial,stripe))
        cj.setOutputPrefix('/u/mhauskn/projects/dct-grad/results/t%d_%d'%(trial,stripe))
        cj.submit()
