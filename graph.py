import random

class Node:
    # ATTRIBUTE
    #
    # output
    # edges, list of edge from Node
    # layer
    # order

    def __init__(self, output = 0, layer = 0, order = 0):
        self.output = output
        self.edges = []
        self.layer = layer
        self.order = order

    def addEdge(self, edge):
        self.edges.append(edge)

class Edge:
    # ATTRIBUTE
    #
    # weight
    # dw, accumulate this through a batch

    def __init__(self):
        self.weight = random.random()
        self.dw = 0
        self.dwBefore = 0

    def updateWeight(self, learningRate):
        self.weight = self.weight + self.dw * learningRate
        self.dw = 0

class Graph:
    # ATTRIBUTE
    # roots, a Node
    # bias, a Node with order = -1
    # children, a graph
    # layer,  indicite current layer

    def __init__(self, bias = Node(), layer = 0):
        self.roots = []
        self.bias = bias
        self.children = None
        self.layer = layer
    
    # Check is this graph an output layer
    def isOutput(self):
        return self.children is None

    # Get the last output of the graph
    def getLastOutput(self):
        graphNow = self
        while not graphNow.isOutput():
            graphNow = graphNow.children
        # print("OUTPUT")
        # print(graphNow.roots[0].output)
        return graphNow.roots[0].output

    # Root is a Node
    # assign it with appropiate order and layer
    def addRoot(self, root):
        root.layer = self.layer
        root.order = len(self.roots)
        self.roots.append(root)

    # Bias is a Node
    # assign it with appropiate order and layer
    def addBias(self, bias):
        bias.layer = self.layer
        bias.order = -1
        self.bias = bias

    # use this to update layer, instead of directly change its attribute
    def updateLayer(self, layer):
        self.layer = layer
        for root in self.roots:
            root.layer = layer

        self.bias.layer = layer

    # Child is a graph
    # assign it with appropiate order and layer
    def addChild(self, children):
        children.updateLayer(self.layer + 1)
        self.children = children

        # add edge from node
        for currentRoot in self.roots:
            for childRoot in children.roots:
                currentRoot.addEdge(Edge())
        
        # add edge from bias
        for childRoot in children.roots:
            self.bias.addEdge(Edge())

    def updateDw(self, learningRate):
        for edge in self.bias.edges:
            edge.updateWeight(learningRate)
        for root in self.roots:            
            for edge in root.edges:
                edge.updateWeight(learningRate)
        if (self.children):
            self.children.updateDw(learningRate)

    def printGraph(self):
        if self.children is None:
            # output layer
            for currentRoot in self.roots:
                print(currentRoot.output, '-> output')
        else:
            for currentRoot in self.roots:
                i = 0
                for childRoot in self.children.roots:
                    print(str(currentRoot.layer) + '.' + str(currentRoot.order) + ' (' + str(currentRoot.output) 
                        + ') --(' + str(currentRoot.edges[i].weight) + ' | ' + str(currentRoot.edges[i].dw) + ')--> ' 
                        + str(childRoot.layer) + '.' + str(childRoot.order) + ' (' + str(childRoot.output) + ')')
                    i = i + 1
            
            i = 0
            for childRoot in self.children.roots:
                print(str(self.bias.layer) + '.' + str(self.bias.order) + ' (' + str(self.bias.output) 
                    + ') --(' + str(self.bias.edges[i].weight) + ' | ' + str(self.bias.edges[i].dw) +')--> ' 
                    + str(childRoot.layer) + '.' + str(childRoot.order) + ' (' + str(childRoot.output) + ')')
                i = i + 1

            self.children.printGraph()


    def printModel(self):
        if self.children is not None:
            for currentRoot in self.roots:
                i = 0
                for childRoot in self.children.roots:
                    print(str(currentRoot.layer) + '.' + str(currentRoot.order) 
                        + ' --(' + str(currentRoot.edges[i].weight) + ')--> ' 
                        + str(childRoot.layer) + '.' + str(childRoot.order))
                    i = i + 1
            
            i = 0
            for childRoot in self.children.roots:
                print(str(self.bias.layer) + '.' + str(self.bias.order) 
                    + ' --(' + str(self.bias.edges[i].weight) + ')--> ' 
                    + str(childRoot.layer) + '.' + str(childRoot.order))
                i = i + 1

            self.children.printModel()
