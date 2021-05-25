"""
	This module contains all code specifically necessary code for the LSTM model
"""
# disable tensorflow logging
import os

from tqdm import tqdm

from preprocessing.sliding_window import prepare_window_off_by_1
from utils.keras import KerasTrainCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential, load_model
from keras.layers import Dense, TimeDistributed
from keras.layers import LSTM

from utils.logger import Logger
from utils.timer import Timer
from configparser import ConfigParser

logger = Logger("LSTM")

tf.random.set_seed(42)
np.random.seed(42)

class LSTMModel:
	"""
		This class describes a LSTM model.

		The LSTM model requires a different approach to train it. This includes a non standard
		prepare function and a fit function that does not take pandas DataFrames but numpy ndarrays.
	"""

	def __init__(self, vars: int, model: Sequential = None):
		"""
			This constructs a LSTMModel for training and usage in a sort of sklearn compatible way.

			Parameters:

				vars: How many input variables there are
		"""

		if model is None:
			# initialize model
			self.model = Sequential()
			self.model.add(LSTM(units=128, activation='sigmoid', input_shape=(None, vars), return_sequences=True))
			self.model.add(TimeDistributed(Dense(vars)))
			self.model.compile(loss='mean_squared_error', optimizer='adam')
			self.model.summary()
		else:
			self.model = model

	def fit(self, y: np.ndarray, x: np.ndarray, epochs: int):
		print(f"Shape: {x.shape}")
		print(f"Model: {self.model.input_shape}")
		with tqdm(total=epochs) as progress:
			nntc = KerasTrainCallback(progress)
			self.model.fit(x=x, y=y, use_multiprocessing=True, callbacks=[nntc], epochs=epochs, verbose=0)
		return self.model

	def predict_next(self, start: np.ndarray) -> np.ndarray:
		assert len(start.shape) == 2
		past: np.ndarray = start.reshape((1, *start.shape))
		y_pred = self.model.predict(past)
		return y_pred[0, -1, :]

	def predict_sequence(self, start: np.ndarray, length: int) -> np.ndarray:
		assert len(start.shape) == 2
		past: np.ndarray = start
		predicted = []
		for _ in range(length):
			y_pred = self.predict_next(past)
			predicted.append(y_pred)
			past = np.append(past, y_pred.reshape((1, -1)), axis=0)
		return np.array(predicted)

	def store(self, config: ConfigParser):
		path = config["storage_location"] + "/autoarima.pkl"
		if not os.path.exists(path):
			self.model.save(path)

	def predict(self, x, fh):
		return self.predict_next(x)

def train_or_load_LSTM(config: ConfigParser, data: pd.DataFrame) -> LSTMModel:
	p = config["storage_location"] + "/autoarima.pkl"
	if os.path.exists(p):
		return load(config)
	else:
		return train_lstm_model(data)



def load(config: ConfigParser) -> LSTMModel:
	path = config["storage_location"] + "/autoarima.pkl"
	return LSTMModel(vars=0, model=load_model(path))

def train_lstm_model(data: pd.DataFrame) -> LSTMModel:
	logger.info("Training LSTM...")
	timer = Timer()
	model = LSTMModel(vars=3)

	# extract the data with a sliding window of length 20
	x_train, y_train = prepare_window_off_by_1(data, 20)

	model.fit(y=y_train, x=x_train, epochs=10)

	logger.info(f"Done in {timer}")
	return model

def test_lstm_model():
	n = 100000
	start = 0
	stop = 140
	# produce some test data: X contains sin, cos and y contains 1/(1+cos^2). Random noise is also added to both
	data_sin = np.sin(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)
	data_cos = np.cos(np.linspace(start, stop, n)) + (np.random.random(n) * 0.4 - 0.2)

	data_res = 1.0 / (1.0 + np.cos(np.linspace(start, stop, n)) ** 2) + (np.random.random(n) * 0.2 - 0.1)
	data = pd.DataFrame([data_sin, data_cos, data_res], index=["sin", "cos", "res"]).transpose()

	# For fitting, we need need x.shape == [n, v, f] and y.shape == [n, v, f] where:
	# - n is the number of samples
	# - v is the number of feature vectors in each sample
	# - f is the number of features
	# The sequence y[i, :-1, :] should equal x[i, 1:, :], i.e. is offset by one

	model = train_lstm_model(data)

	x_test, y_test = prepare_window_off_by_1(data[-50:], 20)
	y_pred = np.array([model.predict_next(x_test[i]) for i in range(x_test.shape[0])])  # model.predict(x_test, fh=10)
	y_pred_seq = model.predict_sequence(x_test[0], y_test.shape[0])

	plt.plot(y_test[:, -1, 0])  # blue
	plt.plot(y_pred[:, 0])  # green
	plt.plot(y_pred_seq[:, 0])  # orange
	plt.show()


if __name__ == '__main__':
	test_lstm_model()
