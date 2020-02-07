import pandas as pd
import numpy as np 
from nn import NeuralNetwork

class MultiLayerPerceptron:
    # ATTRIBUTE
    # 
    # df, full dataframe
    # target, target attribute
    
    # constructor
    def __init__(self, filename, target):
        self.readCsv(filename, target)

    # read file csv with given filename in folder data
    def readCsv(self, filename, target):
        self.df = pd.read_csv('data/' + filename + '.csv')
        self.target = target

    def isContinuous(self, attr):
        return not ((self.df[attr].dtypes == 'bool') or (self.df[attr].dtypes == 'object'))

    def uniqueTargetValue(self):
        self.unique = self.df[self.target].unique().tolist()

    #assign neural network in self attribute
    def assignNeuralNetwork(self):
        self.nn = []
        for u in self.unique:
            print(u)
            newdf = self.df.copy()
            newdf.loc[newdf[self.target]!=u, self.target] = 0
            newdf.loc[newdf[self.target]==u, self.target] = 1
            new_nn = NeuralNetwork(newdf)
            self.nn.append(new_nn)

    # def infoGain(self, df, attr, entropy):
    #     #dataframe row
    #     row = df.shape[0] 

    #     #get unique value
    #     unique = df[attr].unique().tolist()

    #     gain = entropy
    #     for u in unique:
    #         #get occurence
    #         freq = (df[attr]==u).sum()
    #         newdf = self.splitHorizontalKeepValue(df, attr, u)
    #         e = self.entropy(newdf, self.target)
    #         gain -= freq/row * e
        
        # return gain

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

    #handling missing value: get the most frequent value with the same target
    def handling_missing_value(self):
        if self.df.isnull().values.any():
         missing_columns = self.df.columns[self.df.isna().any()].tolist()
        for col in missing_columns:
            mode = self.df.mode()[col][0]
            rows = pd.isnull(self.df).any(1).nonzero()[0].tolist()
            self.df[col][rows]=mode

    def getAllTreshold(self, df, attr):
        sortedDf = self.sortValue(df,attr)
        listClass = sortedDf[self.target].values
        listCandidateForC = []
        i = 0
        while(i < len(listClass)-1):
            if(listClass[i] != listClass[i+1]):
                listAttr = sortedDf[attr].values
                listCandidateForC.append((listAttr[i]+listAttr[i+1])/2)
            i += 1
        return listCandidateForC

    def infoGainContinuous(self, df, attr, entropy, treshold):
        sortedDf = self.sortValue(df,attr)
        row = sortedDf.shape[0]
        gain = entropy
        dfLessThanTreshold = sortedDf[sortedDf[attr] < treshold]
        dfGreaterThanTreshold = sortedDf[sortedDf[attr] >= treshold]
        less = self.entropy(dfLessThanTreshold, self.target)
        greater = self.entropy(dfGreaterThanTreshold, self.target)
        gain -= (dfLessThanTreshold.shape[0]/row * less + dfGreaterThanTreshold.shape[0]/row * greater)
        return gain

    def getBestTreshold(self, df, attr):
        candidate = self.getAllTreshold(df,attr)
        gains = []
        for value in candidate:
            gains.append(self.infoGainContinuous(df, attr, self.entropy(df, self.target), value))
        
        #get index of max attributes
        maxindex = 0
        for i in range (1,len(gains)):
            if (gains[maxindex]<gains[i]):
                maxindex = i

        return candidate[maxindex]

    def makeDiscrete(self, df):
        newDf = df.copy()
        for col in newDf.columns:
            if(self.isContinuous(col)):
                treshold = self.getBestTreshold(df,col)
                row = newDf[col].shape[0]
                for i in range(0, row):
                    if(newDf[col][i] < treshold):
                        newDf[col][i] = LOW
                    else:
                        newDf[col][i] = HIGH
        return newDf