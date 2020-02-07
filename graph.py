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
    def __init__(self):
        self.roots = []
    
    def isOneElement(self):
        return len(self.roots) == 1

    # Root is a Node
    def addRoot(self, root):
        self.roots.append(root)

    # Child is a graph
    def addChild(self, children):
        self.children = children
        for currentRoot in self.roots:
            for childRoot in children.roots:
                currentRoot.addEdge(Edge(0))

    def __str__(self, level=0, value=''):
        if (level == 0):
            ret = repr(self.name)+"\n"
        else:
            ret = "   "*(level-1)+"|--"+repr(self.name)+" ("+ str(value) +")\n"
        for child in self.children:
            ret += child[1].__str__(level=level+1, value=child[0])
        return ret

    def __repr__(self):
            return '<tree node representation>'
        

grapha = Graph()
graphb = Graph()
graph = Graph()
graph = Graph()
graph = Graph()
graph = Graph()
graph = Graph()
graph.addRoot()
# tree = Tree('afd')
# treeb = Tree('bfdsaf')
# treec = Tree('cdasdas')
# treed = Tree('ddasdas')
# treee = Tree('edasdas')
# treef = Tree('fdasdas')
# treeg = Tree('gdasdas')
# treeh = Tree('hdasdas')
# treei = Tree('i')
# treej = Tree('j')
# treek = Tree('k')
# treel = Tree('l')
# treem = Tree('m')
# treen = Tree('n')
# treeo = Tree('o')
# treep = Tree('p')
# treeq = Tree('q')

# tree.addChild('b-val',treeb)
# tree.addChild('c-val',treec)
# treeb.addChild('d-val',treed)
# treec.addChild('f-val',treef)
# treec.addChild('g-val',treeg)
# treec.addChild('h-val',treeh)
# treed.addChild('i-val',treei)
# tree.addChild('wakgeng', treej)
# tree.addChild('coba', treek)
# treej.addChild('cobalagi', treel)
# treeb.addChild('cobaocba', treem)

# print(tree)