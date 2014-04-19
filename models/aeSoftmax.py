# Trains a standard autoencoder followed by a softmax classifier
model = Model()

l1 = Layer(visibleSize, hiddenSize)
model.addLayer(l1)
l2 = Layer(hiddenSize, visibleSize, activation=None)
model.addLayer(l2)
model.finalize()

output = model.forward(x)
cost = reconstructionCost(l2, x) + beta * sparsityCost(l1, spar) + Lambda * (weightCost(l1) + weightCost(l2))
opttheta = train() 
model.setTheta(opttheta)

classifier = Softmax(visibleSize,10)
model.addLayer(classifier)
model.finalize()
        
output = model.forward(x)
cost = xentCost(classifier, y)
accuracy = accuracy(classifier,y)
opttheta = train()        

fname = path+'/results/'+outputPrefix
model.saveImages(fname, opttheta)
model.save('results/out.mdl')

