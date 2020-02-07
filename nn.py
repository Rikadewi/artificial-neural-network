import pandas as pd
import numpy as np 
import math
class NeuralNetwork:
    # ATTRIBUTE
    # 
    # df, full dataframe encoded binomial target

    # constructor
    def __init__(self, df):
        self.df = df

    #List Data yang masuk berupa array feature [xo,x1...xn] 
    #dan list Weight [w0,w1...wn]

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