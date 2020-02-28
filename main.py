from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    nHiddenLayer = 1, 
    nNode = 2, 
    maxIteration=1
)

mlp.printModel()
