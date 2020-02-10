class Node:
    def __init__(self, output = 0):
        self.output = output
        self.edge = []
    def addEdge(self, edge):
        self.edge.append(edge)

class Edge:
    def __init__(self, weight = 0):
        self.weight = weight
        self.dw = 0
    def addDw(self, dw):
        self.dw = self.dw + dw
    def updateWeight(self):
        self.weight = self.weight + self.dw
        self.dw = 0

class Graph:
    def __init__(self, bias = Node()):
        self.roots = []
        self.bias = bias
        self.children = None
    
    def isOneElement(self):
        return len(self.roots) == 1

    # Root is a Node
    def addRoot(self, root):
        self.roots.append(root)

    # Bias is a Node
    def addBias(self, bias):
        self.bias = bias

    # Child is a graph
    def addChild(self, children):
        self.children = children

        # add edge from node
        for currentRoot in self.roots:
            for childRoot in children.roots:
                currentRoot.addEdge(Edge())
        
        # add edge from bias
        for childRoot in children.roots:
            self.bias.addEdge(Edge(0))

    def printGraph(self, level=0, value=''):
        if self.children is None:
            # output layer
            for currentRoot in self.roots:
                print(currentRoot.output, '-> output')
        else:
            for currentRoot in self.roots:
                i = 0
                for childRoot in self.children.roots:
                    print(currentRoot.output, '->', childRoot.output, '(', currentRoot.edge[i].weight, ')')
                    i = i + 1
            
            i = 0
            for childRoot in self.children.roots:
                print(self.bias.output, '->', childRoot.output, '(', self.bias.edge[i].weight, ')')
                i = i + 1

            self.children.printGraph()


grapha = Graph()
graphb = Graph()
graphc = Graph()
graphd = Graph()
graphe = Graph()
graphf = Graph()
graphg = Graph()
roota = Node(1)
rootb = Node(2)
rootc = Node(3)
rootd = Node(4)
roote = Node(5)
rootf = Node(6)
rootg = Node(7)
rooth = Node(8)
rooti = Node(9)
rootj = Node(10)
rootk = Node(11)
rootl = Node(12)
rootm = Node(13)
rootn = Node(14)
rooto = Node(15)
rootp = Node(16)
rootq = Node(17)
rootr = Node(18)
roots = Node(19)
grapha.addRoot(roota)
grapha.addRoot(rootb)
grapha.addRoot(rootc)
grapha.addRoot(rootd)
grapha.addRoot(roote)
graphb.addRoot(rootf)
graphb.addRoot(rootg)
graphb.addRoot(rooth)
graphc.addRoot(rooti)
graphc.addRoot(rootj)
graphc.addRoot(rootk)
graphd.addRoot(rootl)
graphd.addRoot(rootm)
graphd.addRoot(rootn)
graphe.addRoot(rooto)
grapha.addBias(rootp)
graphb.addBias(rootq)
graphc.addBias(rootr)
graphd.addBias(roots)

grapha.addChild(graphb)
graphb.addChild(graphc)
graphc.addChild(graphd)
graphd.addChild(graphe)

grapha.printGraph()
