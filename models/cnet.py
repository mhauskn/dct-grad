model = Model()
model.addLayer(Reshape(inputShape=(batch_size,visibleSize), outputShape=(batch_size,1,28,28)))
model.addLayer(Convolve(inputShape=model.getOutputShape(), filterShape=(20,1,5,5)))
model.addLayer(Pool(inputShape=model.getOutputShape(), poolsize=(2,2)))
# TODO: Add tanh somewhere here
model.addLayer(Convolve(inputShape=model.getOutputShape(), filterShape=(50,20,5,5)))
model.addLayer(Pool(inputShape=model.getOutputShape(), poolsize=(2,2)))
model.addLayer(Reshape(inputShape=model.getOutputShape(),
                       outputShape=(batch_size, np.prod(model.getOutputShape()[1:]))))
model.addLayer(Layer(inputShape=model.getOutputShape(),outputShape=500))
classifier = Softmax(inputShape=model.getOutputShape(),nClasses=10)
model.addLayer(classifier)
model.finalize()

output = model.forward(x)
cost = xentCost(classifier, y)
accuracy = classifier.accuracy(y)
opttheta = train(800)

