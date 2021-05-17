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
	
	def __init__(self, inputs, outputs):
		# initialize model
		self.model = Sequential()
		self.model.add(LSTM(units=10, actiation='sigmoid', input_shape=(inputs, outputs)))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
	
	def fit(self, y, x):
		self.model.fit(x=x, y=y)
		return self.model
	
	def predict(self, x):
		return pd.DataFrame(self.model.predict(x=x))


def train_lstm_model(y: pd.DataFrame, x: pd.DataFrame) -> LSTMModel:
	model = LSTMModel()
	

