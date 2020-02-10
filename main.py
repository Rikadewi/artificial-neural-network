# from mlp import MultiLayerPerceptron

# myMlp = MultiLayerPerceptron("iris", "species")
# # myMlp.uniqueTargetValue()
# myMlp.assignNeuralNetwork(2,3)
# print(myMlp.unique)
# print(myMlp.nn[0].readDf())

from nn import NeuralNetwork
from mlp import MultiLayerPerceptron
from graph import *

myMlp = MultiLayerPerceptron("iris", "species")

net = NeuralNetwork(myMlp.df, myMlp.target)

graph = Graph() # graph coba coba

#panggil method
net._backPropagation(graph)
