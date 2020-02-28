from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    learningRate = 0.2,
    nHiddenLayer = 3, 
    nNode = 10, 
    maxIteration=200
)

for nn in mlp.nn:
    nn.graph.printGraph()