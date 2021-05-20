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
from keras.layers import LSTM
from keras.layers import Dropout

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("LSTM")

tf.random.set_seed(42)
np.random.seed(42)


class LSTMModel:
	"""
		This class describes a LSTM model.

		The LSTM model requires a different approach to train it. This includes a non standard
		prepare function and a fit function that does not take pandas DataFrames but numpy ndarrays.
	"""
	
	def __init__(self, vars: int):
		"""
			This constructs a LSTMModel for training and usage in a sort of sklearn compatible way.
			
			Parameters:
				
				vars: How many input variables there are
		"""
		# set member variables
		self.vars = vars
		
		# initialize model
		self.model = Sequential()
		self.model.add(LSTM(units=10, activation='sigmoid', input_shape=(None, self.vars), return_sequences=False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(self.vars))
		self.model.compile(loss='mean_squared_error', optimizer='adam')
		self.columns = []
	
	def fit(self, x: np.ndarray):
		print(f"\n\nShape: {x.shape}\n\n")
		print(f"Model: {self.model.input_shape}\n\n")
		self.model.fit(x=x)
		return self.model
	
	def predict(self, x: np.ndarray, fh: int) -> np.ndarray:
		y_pred = np.ndarray(self.vars)
		for i in range(fh):
			y_pred = np.append(y_pred, self.model.predict(np.append(arr=x, values=y_pred, axis=0)))
		return y_pred


def train_lstm_model(y: pd.DataFrame, x: pd.DataFrame, fh: int) -> LSTMModel:
	logger.info_begin("Training LSTM...")
	timer = Timer()
	model = LSTMModel(vars=3)
	model.fit(y, x)
	
	logger.info_end(f"Done in {timer}")
	return model


def prepare_dataset_lstm(dataframes: List[pd.DataFrame]) -> np.ndarray:
	"""
		This function reformats a set of DataFrames into a 3-d numpy array that tensorflow wants for its networks.
	"""
	logger.info_begin("Preparing dataset...")
	timer = Timer()

	out = [x.to_numpy() for x in dataframes]

	logger.info_end(f"Done in {timer}")
	return np.array(out)


def test_lstm_model():
	n = 1000
	start = 0
	stop = 70
	# produce some test data: X contains sin, cos and y contains 1/(1+cos^2). Random noise is also added to both
	data_sin = np.sin(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)
	data_cos = np.cos(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)
	
	data_res = 1.0 / (1.0 + np.cos(np.linspace(start, stop, n)) ** 2) + (np.random.random(n) * 0.2 - 0.1)
	
	data = pd.DataFrame([data_sin, data_cos, data_res], index=["sin", "cos", "res"]).transpose()
	
	x_train = prepare_dataset_lstm([data[:-50]])
	x_test = prepare_dataset_lstm([data[-50:]])
	
	model = LSTMModel(vars=3)
	logger.info_begin("Training model...")
	timer = Timer()
	model.fit(x_train)
	logger.info_end(f"Done in {timer}")
	logger.info_begin("Getting prediction...")
	timer = Timer()
	x_pred = model.predict(x_test, fh=10)
	logger.info_end(f"Done in {timer}")
	
	plt.plot(x_train)
	plt.plot(x_pred)
	plt.show()
	

if __name__ == '__main__':
	test_lstm_model()
