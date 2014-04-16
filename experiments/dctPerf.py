# Test the classification accuracy of various dct algorithms
import cluster

j, expDir = cluster.getEnv()

#--------------------- Direct Autoencoder ---------------------#
# pre = 'direct'
# j.setArgs('%s/train.py --nEpochs 200 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# jid = j.submit()

# j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %sII --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
# j.depends(jid)
# j.submit()

#--------------------- Rectangle AE ---------------------#
# pre = 'rectangle'
# j.setArgs('%s/train.py --autoencoder rectangle --nEpochs 200 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# jid = j.submit()

# j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %sII --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
# j.depends(jid)
# j.submit()

# #--------------------- Stripe AE ---------------------#
# pre = 'stripe'
# j.setArgs('%s/train.py --autoencoder stripe --nEpochs 200 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# jid = j.submit()

# j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %sII --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
# j.depends(jid)
# j.submit()

# #--------------------- Reshape AE ---------------------#
# pre = 'reshape'
# j.setArgs('%s/train.py --autoencoder reshape --nEpochs 200 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%s'%(expDir,pre))
# jid = j.submit()

# j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %sII --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
# j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
# j.depends(jid)
# j.submit()

#--------------------- Oddly tuned Reshape AE ---------------------#
pre = 'reshapeExperimental'
j.setArgs('%s/train.py --autoencoder reshape --nEpochs 200 --Lambda .15 --path %s --outputPrefix %s --save %s/results/%s.mdl'%(expDir,expDir,pre,expDir,pre))
j.setOutputPrefix('%s/results/%s'%(expDir,pre))
jid = j.submit()

j.setArgs('%s/train.py --nEpochs 2000 --path %s --outputPrefix %sII --load %s/results/%s.mdl --classify'%(expDir,expDir,pre,expDir,pre))
j.setOutputPrefix('%s/results/%sII'%(expDir,pre))
j.depends(jid)
j.submit()
