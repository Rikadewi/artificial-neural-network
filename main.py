from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron(
    nHiddenLayer = 1, 
    nNode = 2, 
    maxIteration=1
)

for nn in mlp.nn:
    nn.graph.printGraph()
print("accuracy:", mlp.accuration())