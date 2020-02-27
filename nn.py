import pandas as pd
import numpy as np 
import math
from graph import *

class NeuralNetwork:
    # ATTRIBUTE
    # petal_length
    # nHiddenLayer, default = 1 number of hidden layer in one NN
    # nNode, default = 1 number of node in hidden layer 
    # graph, translated graph from number of features, hiddenlayer and node

    # constructor
    def __init__(self, nFeature, nHiddenLayer = 1, nNode = 1):
        self.nHiddenLayer = nHiddenLayer
        self.nNode = nNode
        self.makeStucture(nFeature)

    #List Data yang masuk berupa array feature [xo,x1...xn] 
    #dan list Weight [w0,w1...wn]

    def makeSingleGraph(self, nRoot):
        graph = Graph()
        graph.addBias(Node(1))
        for i in range (0, nRoot):
            graph.addRoot(Node())
        return graph
        
    def makeStucture(self, nFeature):
        #brp banyak feature, hidden layer, unit di hidden layer

        graphs= []
        #buat graph pertama untuk input layer
        graph = self.makeSingleGraph(nFeature)
        graphs.append(graph)

        #buat graph untuk hidden layer
        for i in range (0, self.nHiddenLayer):
            graph = self.makeSingleGraph(self.nNode)
            graphs[len(graphs)-1].addChild(graph)
            graphs.append(graphs[len(graphs)-1].children)

        #buat graph untuk output layer
        graph = self.makeSingleGraph(1)
        graphs[len(graphs)-1].addChild(graph)
        self.graph = graphs[0]
        # graphs[0].printGraph()        

    def sigmaFunction(self, listData, listWeight):
        sum = 0
        for i in range (0, len(listData)):
            sum += listData[i]*listWeight[i]
        return sum

    def sigmoidFunction(self, data):
        return 1/(math.exp(-data)+1)
        
    def deltaW(self, learningRate, target, output, input):
        return (target-output)*(1-output)*output*input*learningRate

    def errorValue(self, target, output):
        return 0.5*pow((target-output),2)

    # x (array of integer), array of feature datas
    def feedForward(self, x):
        # ngisi output (root) dari sebuah graph
        # ngisi x ke input layer    
        for i in range (0, len(x)):
            self.graph.roots[i].output = x[i]

        #logic: loop semua root yang ada di dalam satu graph, kalikan dengan output
        graphNow = self.graph
        nextGraph = self.graph.children
        while nextGraph != None:
            for i in range (0, len(nextGraph.roots)):
                result = 0
                for j in range (0, len(graphNow.roots)):
                    # print(graphNow.roots[j].edges[i].weight)
                    # print(graphNow.roots[j].output)
                    result+=graphNow.roots[j].edges[i].weight*graphNow.roots[j].output
                result+=graphNow.bias.output*graphNow.bias.edges[i].weight
                nextGraph.roots[i].output = result
                print("RESULT = " + str(result))
            graphNow = nextGraph
            nextGraph = nextGraph.children

        return self.graph

    def updateAllDw(self):
        if self.graph:
            self.graph.updateDw()

    def getGraphOutput(self):
        if (self.graph):
            return self.graph.getLastOutput()

    # y (integer), target predict data
    def backPropagation(self, y):
        self.graph = self._backPropagation(self.graph, y)

    # private method of back propagation
    def _backPropagation(self, graph, y):
        # add recursive somwhere, return graph
        if not graph.isOutput():
            for root in graph.roots:
                i = 0
                for edge in root.edges:
                    # all dw before
                    sumDwChild = edge.dw
                    if graph.children.isOutput():
                        sumDwChild = (y - graph.children.roots[i].output)*graph.children.roots[i].output
                    else:    
                        childGraph = self._backPropagation(graph.children, y)
                        for edgeChild in childGraph.roots[i].edges:
                            sumDwChild = sumDwChild + edgeChild.dw*edgeChild.weight

                    edge.dw = sumDwChild*root.output*(1 - graph.children.roots[i].output)
                    i = i + 1

            # update bias
            i = 0
            for edge in graph.bias.edges:
                # all dw before
                sumDwChild = edge.dw
                if graph.children.isOutput():
                    sumDwChild = (y - graph.children.roots[i].output)*graph.children.roots[i].output
                    # print(sumDwChild)
                else:
                    for edgeChild in graph.children.roots[i].edges:
                        sumDwChild = sumDwChild + edgeChild.dw*edgeChild.weight

                edge.dw = sumDwChild*graph.bias.output*(1 - graph.children.roots[i].output)
                i = i + 1

        return graph
