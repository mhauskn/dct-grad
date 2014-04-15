# Test the reconstruction of different Autoencoders
import cluster

j, expDir = cluster.getEnv()

# Part I - Train autoencoders
# pre = 'noTrain'
# j.setArgs('%s/train.py --nEpochs 0 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# j.submit()

# pre = 'trained'
# j.setArgs('%s/train.py --nEpochs 200 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# j.submit()

# pre = 'unregularized'
# j.setArgs('%s/train.py --nEpochs 200 --beta 0 --Lambda 0 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# j.submit()

# Part II - Train a classifier on top
pre = 'noTrain'
j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %s --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
j.submit()

pre = 'trained'
j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %s --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
j.submit()

pre = 'unregularized'
j.setArgs('%s/train.py --nEpochs 2000 --beta 0 --Lambda 0 --path %s --outputPrefix %s --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
j.submit()
