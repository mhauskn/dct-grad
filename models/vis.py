# Trains a visualization layer
model = Model()

def showPart1():
    W0 = model.layers[0].W.eval() # 784x196
    b0 = model.layers[0].b.eval() # 196
    a0 = W0
    W1 = model.layers[1].W.eval() # 196x10
    b1 = model.layers[1].b.eval() # 10
    a1 = W1

    W0_v = a0
    W1_v = np.dot(a1.T, W0_v.T).T
    
    shownet.make_filter_fig(fname='results/l0.png',
                            filters = W0_v,
                            _title='Layer 0 Weights')
    shownet.make_filter_fig(fname='results/l1.png',
                            filters = W1_v,
                            _title='Layer 1 Weights')
    
    # Plotting confusion matrix
    p_y_given_x = model.forward(test_set_x)
    ypred = T.argmax(p_y_given_x, axis=1)
    shownet.plot_confusion_matrix(fname='results/confusion.png',
                                  y_true=test_set_y.eval(),
                                  y_pred=ypred.eval(),
                                  labels=None)

def showPart2():
    W2 = model.layers[2].W.eval() # 10x784
    b2 = model.layers[2].b.eval() # 784
    a2 = W2 + b2

    W2_v = a2.T
    
    shownet.make_filter_fig(fname='results/l2.png',
                            filters = W2_v,
                            _title='Layer 2 Weights')

if args.load:
    model.load(args.load)

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

else:
    inputSz = visibleSize
    outputSz = nHidden
    encode = Layer(inputSz, outputSz, activation=actFn)
    model.addLayer(encode)
    classifier = Softmax(model.getOutputShape(),10)
    model.addLayer(classifier)
    model.finalize()
        
    plotProgress = showPart1

    output = model.forward(x)
    cost = xentCost(classifier, y)
    accuracy = classifier.accuracy(y)
    opttheta = train(100)
    model.setTheta(opttheta)

    # # Add decoding/visualization layer
    # decode = Layer(10, visibleSize, activation=actFn)
    # model.addLayer(decode)

    # # Freeze previous layers
    # model.freezeLayer(0)
    # model.freezeLayer(1)
    # model.finalize()

    # plotProgress = showPart2

    # # Train the decoding layer to reconstruct
    # output = model.forward(x)
    # cost = reconstructionCost(decode, x)
    # opttheta = train(10)

    # # model.unfreezeLayer(0)
    # # model.unfreezeLayer(1)
    # # model.setTheta(opttheta)
    # model.freezeLayer(2)
    # model.save('results/vis.mdl')


