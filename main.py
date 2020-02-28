from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    learningRate = 0.2,
    nHiddenLayer = 3, 
    nNode = 3, 
    errorTreshold = 0.3,
    maxIteration=10
)

mlp.printModel()
