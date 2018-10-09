import numpy as np
import pandas as pd 

class LinearRegressionSimulator(object):
	def __init__ (self, Theta, std):
		"""
		Inputs:
			Theta - array of coefficients (nonempty 1xD+1 numpy array)
			Std - standard deviation (float)
		"""

		assert len(Theta) != 0

		self.Theta = Theta
		self.std = std



	def SimPoly(self, XInput):
		"""
		Input: 
			XInput - (Nx1 pandas dataframe)

		Returns:
			outarray - (N-dim Vector)	
		"""
		
		N,L = XInput.shape # A dataframe (NxL Dataframe)

		assert L ==1

		NewXInput = XInput.copy() # Making a copy of Dataframe

		for i in range (1, len(self.Theta)-1): # Making a NxD Datafare of Input
			NewXInput[i] = XInput**(i+1) 


		self.means =self.Theta[0]+np.matmul(NewXInput, self.Theta[1:]) # Calculating mean
		
		
		outarray = self.std*np.random.randn(N)+self.means # Constructing the result array

		
		return outarray