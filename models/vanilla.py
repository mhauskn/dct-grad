def plotProgress():
    W0 = model.layers[0].W.eval() # 784x196
    b0 = model.layers[0].b.eval() # 196
    a0 = W0
    W1 = model.layers[1].W.eval() # 196x784
    b1 = model.layers[1].b.eval() # 784
    a1 = W1

    W0_v = a0
    W1_v = np.dot(a1.T, a0.T).T #a1.T
    
    shownet.make_filter_fig(fname='results/l0.png',
                            filters = W0_v,
                            _title='Layer 0 Weights')
    shownet.make_filter_fig(fname='results/l1.png',
                            filters = W1_v,
                            _title='Layer 1 Weights')


# Trains a standard autoencoder followed by a softmax classifier
model = Model()
inputSz = visibleSize
outputSz = nHidden
encode = Layer(inputSz, outputSz, activation=actFn)
decode = Layer(outputSz, visibleSize, activation=actFn)
model.addLayer(encode)
model.addLayer(decode)
model.finalize()

output = model.forward(x)
cost = reconstructionCost(decode, x) + beta * sparsityCost(encode, spar) + Lambda * (weightCost(encode) + weightCost(decode))
opttheta = train(200)
model.setTheta(opttheta)

if args.save:
    model.save(args.save)
