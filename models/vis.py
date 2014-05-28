# Trains a visualization layer
model = Model()

if args.load:
    model.load(args.load)
    model.freezeLayer(2)

    W0 = model.layers[0].W # 784x196
    b0 = model.layers[0].b # 196
    a0 = W0
    W1 = model.layers[1].W # 196x10
    b1 = model.layers[1].b # 10
    a1 = W1
    W2 = model.layers[2].W # 10x784
    b2 = model.layers[2].b # 784
    a2 = W2 + b2

    W0_v = a0 
    W1_v = np.dot(a1.T, W0_v.T).T
    W2_v = a2.T
    
    shownet.make_filter_fig(fname='results/l0.png',
                            filters = W0_v,
                            _title='Layer 0 Weights')
    shownet.make_filter_fig(fname='results/l1.png',
                            filters = W1_v,
                            _title='Layer 1 Weights')
    shownet.make_filter_fig(fname='results/l2.png',
                            filters = W2_v,
                            _title='Layer 2 Weights')
    exit()
    
else:
    inputSz = visibleSize
    outputSz = nHidden
    encode = Layer(inputSz, outputSz, activation=actFn)
    model.addLayer(encode)
    classifier = Softmax(model.getOutputShape(),10)
    model.addLayer(classifier)
    model.finalize()
        
    output = model.forward(x)
    cost = xentCost(classifier, y)
    accuracy = classifier.accuracy(y)
    opttheta = train(200)
    model.setTheta(opttheta)

    # Add decoding/visualization layer
    decode = Layer(10, visibleSize, activation=actFn)
    model.addLayer(decode)

    # Freeze previous layers
    model.freezeLayer(0)
    model.freezeLayer(1)
    model.finalize()

    # Train the decoding layer to reconstruct
    output = model.forward(x)
    cost = reconstructionCost(decode, x)
    opttheta = train(50)

    model.unfreezeLayer(0)
    model.unfreezeLayer(1)
    model.setTheta(opttheta)
    model.save('results/vis.mdl')


