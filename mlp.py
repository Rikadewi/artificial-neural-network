import pandas as pd
import numpy as np 
from nn import NeuralNetwork

class MultiLayerPerceptron:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    # nHiddenLayer, default = 1 number of hidden layer in one NN
    # nNode, default = 1 number of node in hidden layer 
    
    # constructor
    def __init__(self, filename, target,nHiddenLayer = 1, nNode = 1):
        self.readCsv(filename, target)
        self.nHiddenLayer = nHiddenLayer
        self.nNode = nNode
        self.assignNeuralNetwork()

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

    def splitHorizontalKeepValue(self, df, attr, val):
        newdf = df[df[attr]==val]
        return newdf.reset_index(drop=True)

    # return 2 dataframes, by a treshold value
    def splitHorizontalContinuous(self, df, attr, val):
        lowDf = df[df[attr] < val].reset_index(drop=True)
        highDf = df[df[attr] >= val].reset_index(drop=True)
        return lowDf, highDf

    def splitHorizontalDiscardValue(self, df, attr, val):
        newdf = df[df[attr]!=val]
        return newdf.reset_index(drop=True)
    
    # return 2 dataframes, first dataframe size is given percetage of original set, 
    # and the second is the rest of it
    def splitByPercentage(self, percentage=80):
        idx = (round(self.df.shape[0]*percentage/100))
        return np.split(self.df, [idx])
        
    def dropAttr(self, df, attr):
        return df.drop(columns=attr)

    def sortValue(self, df, attr):
        return df.sort_values(by=[attr])

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

    def miniBatch(self, batchsize=32, errortreshold=0.1):
        maxiteration = 1
        # errortreshold = 0.01
        batches = self.makeBatches(batchsize)
        for i in range(0, len(self.nn)):
            # print("NN [" + str(i) + "]")
            error = 100
            iteration = 0
            while(iteration < maxiteration) and (error > errortreshold):
                j = 0
                idxbatch = 0
                while (error > errortreshold) and (idxbatch < len(batches)):
                    errorlist = []
                    for x in batches[idxbatch]:
                        self.nn[i].feedForward(x)
                        self.nn[i].backPropagation(self.newdf[i][self.target][j])
                        self.nn[i].graph.printGraph()
                        print('-------------------------')
                        # print("ini x")
                        # print(x)
                        # print("ini target")
                        print(self.newdf[i][self.target][j])
                        errorlist.append(self.nn[i].errorValue(self.newdf[i][self.target][j], self.nn[i].getGraphOutput()))
    
                        j += 1
                    idxbatch += 1
                    print('batch baru')
                    self.nn[i].updateAllDw()
                    # print("ERRORLIST " + str(len(errorlist)))
                    # print(errorlist)
                    # print("SUM ERROR")
                    error = np.sum(errorlist)
                    # print(error)
                iteration += 1
                print('nn baru')
                # print()
        print(iteration)

mlp = MultiLayerPerceptron("iris", "species", 1, 1)
mlp.miniBatch()