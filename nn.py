import pandas as pd
import numpy as np 
import math
from graph import *

class NeuralNetwork:
    # ATTRIBUTE
    # petal_length
    # df, full dataframe encoded binomial target
    # target, target attribute
    # nHiddenLayer, default = 1 number of hidden layer in one NN
    # nNode, default = 1 number of node in hidden layer 
    # graph, translated graph from number of features, hiddenlayer and node

    # constructor
    def __init__(self, df, target, nHiddenLayer = 1, nNode = 1):
        self.df = df
        self.nHiddenLayer = nHiddenLayer
        self.nNode = nNode
        self.target = target

    #List Data yang masuk berupa array feature [xo,x1...xn] 
    #dan list Weight [w0,w1...wn]

    def makeGraph(self, nRoot):
        graph = Graph()
        graph.addBias(Node())
        for i in range (0, nRoot):
            graph.addRoot(Node())
        return graph
        
    def readDf(self):
        #brp banyak feature, hidden layer, unit di hidden layer
        #banyak feature
        nFeature = len(self.df.columns)-1

        graphs= []
        #buat graph pertama untuk input layer
        graph = self.makeGraph(nFeature)
        graphs.append(graph)

        #buat graph untuk hidden layer
        for i in range (0, self.nHiddenLayer):
            graph = self.makeGraph(self.nNode)
            graphs[len(graphs)-1].addChild(graph)
            graphs.append(graphs[len(graphs)-1].children)

        #buat graph untuk output layer
        graph = self.makeGraph(1)
        graphs[len(graphs)-1].addChild(graph)
        self.graph = graphs[0]
        graphs[0].printGraph()        

    def sigmaFunction(self, listData, listWeight):
        sum = 0
        for i in range (0, len(listData)):
            sum += listData[i]*listWeight[i]
        return sum

    def sigmoidFunction(self, data):
        return 1/(math.exp(-data)+1)
        
    def deltaW(self, learningRate, target, output, input):
        return (target-output)*(1-output)*output*input*learningRate

    def errorValue(self, target, ouput):
        return 0.5*pow((target-output),2)

    def feedForward(self):
        ##
        pass
    
    def updateAllDw(self):
        if (self.graph):
            self.updateGraphDw(self.graph)
    
    def updateGraphDw(self, graph):
        for root in graph
     :             for edge in root.edges:
                edge.updateWeight()
        if (graph.children):
            pdateGraphDw()
    graph.children    
    def backPropagation(self):
        _backPropagation(self.graph)

    # private method of back propagation
    def _backPropagation(self, graph):
        if graph.children is not None:
            for root in graph.roots:
                for edge in root.edges:
                    edges.dw = 
            
