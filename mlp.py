import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from nn import NeuralNetwork

class MultiLayerPerceptron:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    # nHiddenLayer, default = 1 number of hidden layer in one NN
    # nNode, default = 1 number of node in hidden layer 
    
    # constructor
    def __init__(self, 
            filename='iris', target='species', nHiddenLayer = 4, 
            nNode = 4, batchsize=32, errorTreshold=0.1, maxIteration=10):
        self.readCsv(filename, target)
        self.splitDf()
        # print(self.df)
        # print(self.test)
        # print(self.accuration())

        self.nHiddenLayer = nHiddenLayer
        self.nNode = nNode
        self.assignNeuralNetwork()
        self.miniBatch(batchsize, errorTreshold, maxIteration)

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target

    def isContinuous(self, attr):
        return not ((self.df[attr].dtypes == 'bool') or (self.df[attr].dtypes == 'object'))

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
        # errorTreshold = 0.01
        batches = self.makeBatches(batchsize)
        for i in range(0, len(self.nn)):
            # print("NN [" + str(i) + "]")
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
                        # print('-------------------------')
                        # print("ini x")
                        # print(x)
                        # print("ini target")
                        # print(self.newdf[i][self.target][j])
                        errorlist.append(self.nn[i].errorValue(self.newdf[i][self.target][j], self.nn[i].getGraphOutput()))
    
                        j += 1
                    idxbatch += 1
                    # print('batch baru')
                    self.nn[i].updateAllDw()
                    # self.nn[i].graph.printGraph()
                    # print("ERRORLIST " + str(len(errorlist)))
                    # print(errorlist)
                    # print("SUM ERROR")
                    error = np.sum(errorlist)
                    # print(error)
                iteration += 1
                # print('nn baru')
                # print()
        # print(iteration)

    #x is array of feature data for predict
    def predict(self, x):
        predictCandidate = []
        for nn in self.nn :
            # nn.graph.printGraph()
            output = nn.feedForward(x).getLastOutput()
            predictCandidate.append(output)
        print(predictCandidate)

        predictIndex = predictCandidate.index(max(predictCandidate))
        print(predictIndex)
        # print(self.unique[predictIndex])
        print(self.unique)

        return self.unique[predictIndex]

    def splitDf(self):
        self.df , self.test = train_test_split(self.df, test_size = 0.2)
        # print(self.df)
        self.df = (self.df).reset_index(drop=True)
        # print(self.df)
        self.test = (self.test).reset_index(drop=True)


    def accuration(self):
        hit = 0
        testDf = self.test.copy()
        testDataSet = self.dropAttr(testDf, self.target).values.tolist()
        testValidationSet = testDf[self.target].values.tolist()
        # print(testDataSet)
        # print(testValidationSet)
        for i in range(len(testDataSet)):
            # if("setosa" == testValidationSet[i]):
            if(self.predict(testDataSet[i]) == testValidationSet[i]):
                hit += 1
        return float(hit) / float(len(testValidationSet))
        # print(testData)

    
