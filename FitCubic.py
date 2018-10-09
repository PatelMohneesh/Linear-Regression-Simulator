"""
Created on Fri Feb  3 22:12:27 2017

@author: Mohneesh
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from LinearRegressionSimulator import LinearRegressionSimulator

class LinearRegressionSimulator:
    def __init__(self, Theta, std):
        """
        Inputs:
            Theta - array of coefficients (nonempty 1xD+1 numpy array)
            std   - standard deviation (float)
        """
        
        assert len(Theta) != 0
        
        self.Theta = Theta
        self.std = std
        
    
        
    def SimPoly(self, XInput):
        """
            Input:    
                XInput -(Nx1 pandas dataframe)
            Output:    
                outarray - (N-dim Vector)
        """
        N,D = XInput.shape
        
        NewXInput = XInput.copy()
        
        
        for i in range(1,len(self.Theta)-1):
            
            NewXInput[i]=XInput[0]**(i+1)

        
        self.means = self.Theta[0]+np.matmul(NewXInput, self.Theta[1:])
        
        outarray = self.std*np.random.randn(N)+self.means
        
        return outarray
        
    def FitCubic(self,XInput,y,D):
        
        N,L = XInput.shape
        NXInput=XInput.copy()
        
        for i in range(0,D+1):
            NXInput[i] = XInput[0].copy()**i
            NXInputTran=pd.DataFrame.transpose(NXInput)
            ThetaStar=np.matmul(np.linalg.pinv(np.matmul(NXInputTran,NXInput)),np.matmul(NXInputTran,y)) 
            
        
        np.savetxt('ThetaStar.[Run].[D].txt',ThetaStar)
        
        return ThetaStar

    def Risk(self,xTR,yTR,xTS,yTS,N,M,D):
        
        Degree=[]
        RiskTrain =[]
        RiskTest=[]
        xTR1=xTR.copy()
        xTS1=xTS.copy()
        
        
        for d in range(0,D+1):
            Degree.append(d)
            theta=self.FitCubic(xTR,yTR,d)
            
            
            xTR1[d] = xTR[0].copy()**d
            xTS1[d] = xTS[0].copy()**d

            
            TR_risk = yTR - np.matmul(xTR1, theta)
            
            TR_val = np.sum(np.power(TR_risk,2))
            
            RiskTrain.append(TR_val/(2*N))
            
            TS_risk = yTS - np.matmul(xTS1, theta)
            
            TS_val = np.sum(np.power(TS_risk,2))
            
            RiskTest.append(TS_val/(2*M))
        
        print(RiskTrain)
        print(RiskTest)
        np.savetxt('Risk.train.[Run].txt',RiskTrain,delimiter=' ',newline='\n')
        np.savetxt('Risk.test.[Run].txt',RiskTest,delimiter=' ',newline='\n')
        
        plt.plot(Degree,RiskTrain,'r--',Degree,RiskTest,'b--')
        plt.savefig('RiskPlot.[RUN].pdf')
           
if __name__ == "__main__":
    
    
    Theta=np.array([3,5,4,2])
    std=0.1
    
    obj=LinearRegressionSimulator(Theta,std)
    
    D=10
    N=10
    M=100
    
    TrainingInput = pd.DataFrame(np.random.uniform(0,1,N))
    TrainingOutput=obj.SimPoly(TrainingInput)
    
    np.savetxt('x.train.[Run].txt',TrainingInput,newline='\n')
    np.savetxt('y.train.[Run].txt',TrainingOutput,newline='\n')
    
    TestingInput = pd.DataFrame(np.random.uniform(0,1,M))
    TestingOutput=obj.SimPoly(TestingInput)
    
    np.savetxt('x.test.[Run].txt',TestingOutput,newline='\n')
    np.savetxt('y.test.[Run].txt',TestingOutput,newline='\n')
    
    obj.Risk(TrainingInput,TrainingOutput,TestingInput,TestingOutput,N,M,D)
    print (TrainingInput)
    print (TrainingOutput)
    print (TestingInput)
    print (TestingOutput)

    
  