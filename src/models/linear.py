"""
	This module contains all code specifically necessary code for the LSTM model
"""
from typing import List

# disable tensorflow logging
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("LSTM")

tf.random.set_seed(42)
np.random.seed(42)


class LinearModel:
	"""
		This class describes a LSTM model.

		The LSTM model requires a different approach to train it. This includes a non standard
		prepare function and a fit function that does not take pandas DataFrames but numpy ndarrays.
	"""
	
	def __init__(self, fh: int, train_length: int, train_vars: int):
		"""
			This constructs a LinearModel for training and usage in a sort of sklearn compatible way.
			
			Parameters:
				
				fh: Forecasting horizon for the model. This needs to be specified to the constructor
				for this model, since it actually defines the topology
				
				train_length: The number of timepoints that is the model is trained on per training iteration.
				
				train_vars: How many input variables there are
		"""
		# set member variables
		self.fh = fh
		self.train_length = train_length
		self.train_vars = train_vars
		
		# initialize model
		self.model = Sequential()
		self.model.add(Dense(units=self.train_length * self.train_vars))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(units=self.train_length * self.train_vars))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(fh))
		self.model.compile(loss='mean_squared_error', optimizer='adam')
		self.columns = []
	
	def fit(self, y: np.ndarray, x: np.ndarray):
		print(x.shape, y.shape)
		self.model.fit(x=x, y=y)
		return self.model
	
	def predict(self, x) -> np.ndarray:
		y_pred = self.model.predict(x=x)
		return y_pred


def train_linear_model(y: pd.DataFrame, x: pd.DataFrame, fh: int) -> LinearModel:
	logger.info_begin("Training Linear Network...")
	timer = Timer()
	model = LinearModel(fh=fh)
	model.fit(y, x)
	
	logger.info_end(f"Done in {timer}")
	return model
	

def prepare_dataset_linear(dataframes: List[pd.DataFrame], in_names: List[str], out_names: List[str], unit_size: int, fh: int)\
		-> (np.ndarray, np.ndarray, List[str], List[str]):
	"""
		This function reorganises a set of DataFrames into the 3-d numpy array that tensorflow wants for
		the Linear network.
		
		Parameters:
			
			dataframes: A list containing all the dataframes to be trained on. These dataframes should contain all
			input columns (usually designated X) and all output columns (usually designated y).
			
			in_names: The names of the columns in X, that the network should take as an input.
			
			out_names: The names of the columns in y, that the network should take as output.
			
			unit_size: The amount of timesteps in a single unit. This includes the amount of timesteps used as inputs
			and the amount of timesteps produced as outputs. Since the network will require both in the correct sizes
			when training or testing.
			
			fh: The forecasting horizon. This describes how many timesteps the network will attempt to predict.
		
		Returns:
			
			A tuple containing the input data, output data, columns names of the input and column names of the output.
			The names are included in case they will be needed later on, because the numpy ndarrays cannot store the
			names the columns had when in the pandas dataframe.
		
		Authors:
			linvogel
	"""
	logger.info_begin("Preparing Dataset for LSTM...")
	timer = Timer()
	
	# create lists that contain the finalized data
	in_data: List[np.ndarray] = []
	out_data: List[np.ndarray] = []
	
	# produce dataframes of length unit_size
	dataframes_sized: List[pd.DataFrame] = []
	for df in dataframes:
		length_of_frame: int = df.shape[0]
		
		# split the data into as many disjoint chunks as possible
		for i in range(int(math.floor(length_of_frame / unit_size))):
			dataframes_sized.append(df[i*unit_size:(i+1)*unit_size])
	
	# go through all provided datasets
	for df in dataframes_sized:
		# extract the wanted columns and store them in the output lists
		df_in: pd.DataFrame = df[in_names]
		df_out: pd.DataFrame = df[out_names]
		in_data.append(df_in[:-fh].to_numpy(copy=True))
		out_data.append(df_out[-fh:].to_numpy(copy=True))
		
	logger.info_end(f"Done in {timer}")
	return np.array(in_data), np.array(out_data), in_names, out_names


def test_linear_model():
	n = 1000
	start = 0
	stop = 70
	# produce some test data: X contains sin, cos and y contains 1/(1+cos^2). Random noise is also added to both
	data_sin = np.sin(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)
	data_cos = np.cos(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)
	
	data_res = 1.0 / (1.0 + np.cos(np.linspace(start, stop, n)) ** 2) + (np.random.random(n) * 0.2 - 0.1)
	
	data = pd.DataFrame([data_sin, data_cos, data_res], index=["sin", "cos", "res"]).transpose()
	
	x_train, y_train, _, _ = prepare_dataset_linear([data[:-250]], ["sin", "cos"], ["res"], 250, 50)
	x_test, y_test, _, _, = prepare_dataset_linear([data[-250:]], ["sin", "cos"], ["res"], 250, 50)
	
	print(f"Shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}")
	
	l_train = len(x_train)
	l_test = len(x_test)
	
	model = LinearModel(fh=50, train_length=200, train_vars=2)
	model.fit(y_train.reshape((l_train, 1, 50)), x_train.reshape((l_train, 1, 400)))
	y_pred = model.predict(x_test.reshape((l_test, 1, 400)))
	
	plt.plot(y_test)
	plt.plot(y_pred)
	plt.show()
	

if __name__ == '__main__':
	test_linear_model()
