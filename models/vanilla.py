# Trains a standard autoencoder followed by a softmax classifier
model = Model()
inputSz = visibleSize
outputSz = nHidden
encode = Layer(inputSz, outputSz, activation=actFn)
decode = Layer(outputSz, visibleSize, activation=actFn)
model.addLayer(encode)
model.addLayer(decode)
# model.finalize()

# output = model.forward(x)
# cost = reconstructionCost(decode, x) + beta * sparsityCost(encode, spar) + Lambda * (weightCost(encode) + weightCost(decode))
# opttheta = train(200)
# model.setTheta(opttheta)

# fname = path+'/results/pre'+outputPrefix
# model.saveImages(fname, opttheta)

classifier = Softmax(model.getOutputShape(),10)
model.addLayer(classifier)
model.finalize()
        
output = model.forward(x)
cost = xentCost(classifier, y)
accuracy = classifier.accuracy(y)
opttheta = train(800)

# fname = path+'/results/post'+outputPrefix
# model.saveImages(fname, opttheta)

if args.save:
    model.save(args.save)
