# Trains a standard autoencoder followed by a softmax classifier
model = Model()
layerSizes = [visibleSize, 500, 250, 196]

for i in xrange(len(layerSizes)-1):
    inputSz = layerSizes[i]
    outputSz = layerSizes[i+1]
    encode = Layer(inputSz, outputSz)
    decode = Layer(outputSz, visibleSize, activation=None)
    model.addLayer(encode)
    model.addLayer(decode)
    model.finalize()

    output = model.forward(x)
    cost = reconstructionCost(decode, x) + beta * sparsityCost(encode, spar) + Lambda * (weightCost(encode) + weightCost(decode))
    opttheta = train()
    model.setTheta(opttheta)

    model.deleteLayer() # Remove the top decoding layer
    model.freezeLayer(i)

# Unfreeze all the layers
for i in xrange(len(layerSizes)-1):
    model.unfreezeLayer(i)

classifier = Softmax(model.getOutputSize(),10)
model.addLayer(classifier)
model.finalize()
        
output = model.forward(x)
cost = xentCost(classifier, y)
accuracy = classifier.accuracy(y)
opttheta = train()

fname = path+'/results/'+outputPrefix
model.saveImages(fname, opttheta)
model.save('results/out.mdl')

