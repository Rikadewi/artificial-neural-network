import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from nn import NeuralNetwork

SEED = 13517006

class MultiLayerPerceptron:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    # nHiddenLayer, default = 1 number of hidden layer in one NN
    # nNode, default = 1 number of node in hidden layer 
    
    # constructor
    def __init__(self, 
            filename='iris', target='species', learningRate=0.1, nHiddenLayer = 2, 
            nNode = 10, batchsize=30, errorTreshold=0.3, maxIteration=200):
        self.readCsv(filename, target)
        self.learningRate = learningRate
        self.nHiddenLayer = nHiddenLayer
        self.nNode = nNode
        self.assignNeuralNetwork()
        self.splitDf()
        self.miniBatch(batchsize, errorTreshold, maxIteration)

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target

    #assign neural network in self attribute
    def assignNeuralNetwork(self):
        self.nn = []
        self.newdf = []
        self.unique = self.df[self.target].unique().tolist()
        for u in self.unique:
            newdf = self.df.copy()
            newdf.loc[newdf[self.target]!=u, self.target] = 0
            newdf.loc[newdf[self.target]==u, self.target] = 1
            nFeature = len(self.df.columns)-1
            new_nn = NeuralNetwork(nFeature, self.nHiddenLayer, self.nNode)
            self.nn.append(new_nn)
            self.newdf.append(newdf)

        
    def dropAttr(self, df, attr):
        return df.drop(columns=attr)

    def makeBatches(self, batchsize):
        numofbatches = int(self.df.shape[0]/batchsize)
        remainder = self.df.shape[0]%batchsize
        batches = []
        for i in range (0, numofbatches):
            newbatch = self.dropAttr(self.df, self.target)[i*batchsize: (i+1)*batchsize].values.tolist()
            batches.append(newbatch)
        if (remainder):
            newbatch = self.dropAttr(self.df, self.target)[(i+1)*batchsize : (i+1)*batchsize + remainder].values.tolist()
            batches.append(newbatch)
        return batches

    def miniBatch(self, batchsize, errorTreshold, maxIteration):
        batches = self.makeBatches(batchsize)
        for i in range(0, len(self.nn)):
            error = 100
            iteration = 0
            while(iteration < maxIteration) and (error > errorTreshold):
                j = 0
                idxbatch = 0
                while (error > errorTreshold) and (idxbatch < len(batches)):
                    errorlist = []
                    for x in batches[idxbatch]:
                        self.nn[i].feedForward(x)
                        self.nn[i].backPropagation(self.newdf[i][self.target][j])
                        # self.nn[i].graph.printGraph()
                        errorlist.append(self.nn[i].errorValue(self.newdf[i][self.target][j], self.nn[i].getGraphOutput()))
    
                        j += 1
                    idxbatch += 1
                    self.nn[i].updateAllDw(self.learningRate)
                    error = np.sum(errorlist)
                iteration += 1

    #x is array of feature data for predict
    def predict(self, x):
        predictCandidate = []
        for nn in self.nn :
            output = nn.feedForward(x).getLastOutput()
            predictCandidate.append(output)

        predictIndex = predictCandidate.index(max(predictCandidate))
        return self.unique[predictIndex]

    def splitDf(self):
        self.df , self.test = train_test_split(self.df, test_size = 0.2, random_state=SEED)
        self.df = (self.df).reset_index(drop=True)
        self.test = (self.test).reset_index(drop=True)

    def accuration(self):
        hit = 0
        testDf = self.test.copy()
        testDataSet = self.dropAttr(testDf, self.target).values.tolist()
        testValidationSet = testDf[self.target].values.tolist()
        for i in range(len(testDataSet)):
            if(self.predict(testDataSet[i]) == testValidationSet[i]):
                hit += 1
        return float(hit) / float(len(testValidationSet))

    def printModel(self):
        for nn in self.nn:
            nn.graph.printModel()
        
        print("\naccuracy:", self.accuration())
