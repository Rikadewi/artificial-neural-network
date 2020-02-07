from mlp import MultiLayerPerceptron

myMlp = MultiLayerPerceptron("iris", "species")
myMlp.uniqueTargetValue()
myMlp.assignNeuralNetwork()
print(myMlp.unique)
print(myMlp.nn[2].df)