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

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target

    def isContinuous(self, attr):
        return not ((self.df[attr].dtypes == 'bool') or (self.df[attr].dtypes == 'object'))

    def uniqueTargetValue(self):
        self.unique = self.df[self.target].unique().tolist()

    #assign neural network in self attribute
    def assignNeuralNetwork(self, nHiddenLayer=0, nNode=0):
        self.nn = []
        self.uniqueTargetValue()
        for u in self.unique:
            print(u)
            newdf = self.df.copy()
            newdf.loc[newdf[self.target]!=u, self.target] = 0
            newdf.loc[newdf[self.target]==u, self.target] = 1
            new_nn = NeuralNetwork(newdf, self.target, nHiddenLayer, nNode)
            self.nn.append(new_nn)

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

    