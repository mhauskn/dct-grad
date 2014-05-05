# Trains a standard autoencoder followed by a softmax classifier
model = Model()
visibleSize = 10*10
inputSz = visibleSize
outputSz = 196
encode = Layer(inputSz, outputSz, activation=actFn)
decode = Layer(outputSz, visibleSize, activation=actFn)
model.addLayer(encode)
model.addLayer(decode)
model.finalize()

output = model.forward(x)
cost = reconstructionCost(decode, x) + beta * sparsityCost(encode, spar) + Lambda * (weightCost(encode) + weightCost(decode))
opttheta = train()
model.setTheta(opttheta)

fname = path+'/results/pre'+outputPrefix
model.saveImages(fname, opttheta)

classifier = Softmax(model.getOutputSize(),10)
model.addLayer(classifier)
model.finalize()
        
output = model.forward(x)
cost = xentCost(classifier, y)
accuracy = classifier.accuracy(y)
opttheta = train(10000)

fname = path+'/results/post'+outputPrefix
model.saveImages(fname, opttheta)

if args.save:
    model.save(args.save)
