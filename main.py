from mlp import MultiLayerPerceptron

mlp = MultiLayerPerceptron()
# mlp = MultiLayerPerceptron("iris", "species", 2, 100)
# mlp.miniBatch()

while(True):
    print("Please input " + str(len(mlp.df.columns)-1) + " features for predict")
    for i in range(len(mlp.df.columns)-1):
        x = []
        x.append(float(input()))
    print(mlp.predict(x))

# myMlp = MultiLayerPerceptron("iris", "species", 1, 1)
# # print(myMlp.unique)
# myMlp.nn[0].graph.roots[0].output = 0.1
# myMlp.nn[0].graph.roots[1].output = 0.2
# myMlp.nn[0].graph.roots[2].output = 0.3
# myMlp.nn[0].graph.roots[3].output = 0.4
# myMlp.nn[0].graph.roots[0].addEdge(Edge(0.15))
# myMlp.nn[0].graph.roots[1].addEdge(Edge(0.25))
# myMlp.nn[0].graph.roots[2].addEdge(Edge(0.35))
# myMlp.nn[0].graph.bias.addEdge(Edge(0.45))
# myMlp.nn[0].graph.children.roots[0].addEdge(Edge(0.55))
# myMlp.nn[0].graph.printGraph()

# grapha = Graph()
# graphb = Graph()
# graphc = Graph()
# graphd = Graph()
# graphe = Graph()
# graphf = Graph()
# graphg = Graph()
# roota = Node(1)
# rootb = Node(2)
# rootc = Node(3)
# rootd = Node(4)
# roote = Node(5)
# rootf = Node(6)
# rootg = Node(7)
# rooth = Node(8)
# rooti = Node(9)
# rootj = Node(10)
# rootk = Node(11)
# rootl = Node(12)
# rootm = Node(13)
# rootn = Node(14)
# rooto = Node(15)
# rootp = Node(16)
# rootq = Node(17)
# rootr = Node(18)
# roots = Node(19)
# grapha.addRoot(roota)
# grapha.addRoot(rootb)
# grapha.addRoot(rootc)
# grapha.addRoot(rootd)
# grapha.addRoot(roote)
# graphb.addRoot(rootf)
# graphb.addRoot(rootg)
# graphb.addRoot(rooth)
# graphc.addRoot(rooti)
# graphc.addRoot(rootj)
# graphc.addRoot(rootk)
# graphd.addRoot(rootl)
# graphd.addRoot(rootm)
# graphd.addRoot(rootn)
# graphe.addRoot(rooto)
# grapha.addBias(rootp)
# graphb.addBias(rootq)
# graphc.addBias(rootr)
# graphd.addBias(roots)

# grapha.addChild(graphb)
# graphb.addChild(graphc)
# graphc.addChild(graphd)
# graphd.addChild(graphe)

# nnet = NeuralNetwork(2)
# nnet.graph.roots[0].output = 1
# nnet.graph.roots[0].edges[0].weight = 7
# nnet.graph.roots[1].output = 2
# nnet.graph.roots[1].edges[0].weight = 8
# nnet.graph.bias.output = 5
# nnet.graph.bias.edges[0].weight = 9
# nnet.graph.children.roots[0].output = 3
# nnet.graph.children.roots[0].edges[0].weight = 10
# nnet.graph.children.bias.output = 6
# nnet.graph.children.bias.edges[0].weight = 11
# nnet.graph.children.children.roots[0].output = 4
# nnet.graph.printGraph()

# nnet.backPropagation(5)
# nnet.graph.printGraph()
