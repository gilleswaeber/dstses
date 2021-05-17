"""
	This module contains all code specifically necessary code for the LSTM model
"""
import tensorflow as tf
import pandas as pd
import numpy as np

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
		This class describes a LSTM model
	"""
	
	def __init__(self, fh: int):
		# initialize model
		self.model = Sequential()
		self.model.add(LSTM(units=fh, activation='sigmoid'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.compile(loss='mean_squared_error', optimizer='adam')
		self.columns = []
	
	def fit(self, y: pd.DataFrame, x: pd.DataFrame):
		self.columns = y.columns
		self.model.fit(x=x.to_numpy().reshape((1,) + x.shape), y=y.to_numpy().reshape((1,) + y.shape))
		return self.model
	
	def predict(self, x):
		y_pred = self.model.predict(x=x.to_numpy().reshape((1,) + x.shape))
		return pd.DataFrame(y_pred, columns=self.columns)


def train_lstm_model(y: pd.DataFrame, x: pd.DataFrame, fh: int) -> LSTMModel:
	logger.info_begin("Training LSTM...")
	timer = Timer()
	model = LSTMModel(fh=fh)
	model.fit(y, x)
	
	logger.info_end(f"Done in {timer}")
	return model
	

