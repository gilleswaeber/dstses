"""
	This module contains all code specifically necessary code for the LSTM model
"""
# disable tensorflow logging
import json
import os
from typing import List

from pandas import DataFrame
from tqdm import tqdm

from preprocessing.sliding_window import prepare_window_off_by_1, slide_rows
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
from configparser import ConfigParser, SectionProxy

logger = Logger("LSTM")

tf.random.set_seed(42)
np.random.seed(42)


class LSTMModel:
	"""
		This class describes a LSTM model.

		The LSTM model requires a different approach to train it. This includes a non standard
		prepare function and a fit function that does not take pandas DataFrames but numpy ndarrays.
	"""

	def __init__(self, vars_in: int, vars_out: int, model: Sequential = None):
		"""
			This constructs a LSTMModel for training and usage in a sort of sklearn compatible way.

			Parameters:

				vars_in: How many input variables there are
		"""

		if model is None:
			# initialize model
			self.model = Sequential()
			self.model.add(LSTM(units=128, activation='sigmoid', input_shape=(None, vars_in), return_sequences=True))
			self.model.add(TimeDistributed(Dense(vars_out)))
			self.model.compile(loss='mean_squared_error', optimizer='adam')
			self.model.summary()
		else:
			self.model = model

	def fit(self, y: np.ndarray, x: np.ndarray, epochs: int):
		print(f"Shape: {x.shape}")
		print(f"Model: {self.model.input_shape}")
		with tqdm(total=epochs, desc='LSTM training', dynamic_ncols=True) as progress:
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
		path = config["storage_location"] + "/lstm.pkl"
		if not os.path.exists(path):
			self.model.save(path)

	def predict(self, x, fh):
		return self.predict_next(x)


def train_or_load_LSTM(config: SectionProxy, data: pd.DataFrame) -> LSTMModel:
	p = config["storage_location"] + "/lstm.pkl"
	if os.path.exists(p):
		return load(config)
	else:
		return train_lstm_model_predict(config, data)


def load(config: SectionProxy) -> LSTMModel:
	path = config["storage_location"] + "/lstm.pkl"
	return LSTMModel(vars_in=0, vars_out=0, model=load_model(path))


def split_by_location(df: DataFrame) -> List[DataFrame]:
	"""Split into several dataframes using the the part before the dot in the column name as criterion"""
	df_cols = list(df.columns)
	locations = set(c.split('.')[0] for c in df_cols)
	by_location = []
	for l in locations:
		loc_cols = [c for c in df_cols if c.split('.')[0] == l]
		by_location.append(df[loc_cols].rename(columns={c: c.split('.')[1] for c in loc_cols}))
	return by_location


def split_on_gaps(df: DataFrame, gap_gte_seconds) -> List[DataFrame]:
	chunks = []
	gaps = ((df.index[1:] - df.index[:-1]).seconds >= gap_gte_seconds).nonzero()[0].tolist()
	gaps.append(len(df))
	start = 0
	for gap in gaps:
		chunks.append(df[start:gap])
		start = gap
	return chunks


def train_lstm_model_predict(config: SectionProxy, data: pd.DataFrame) -> LSTMModel:
	"""Several things to do here: split by location, do a gap detection to split in chunks, pass each chunk through the sliding window function, …"""
	logger.info("Preparing LSTM training data...")

	gap_detection = int(config["gap_detection_seconds"])
	stride = int(config["train_window"])
	features_in = json.loads(config["features_in"])
	features_out = json.loads(config["features_out"])
	use_offset = config.getboolean("use_offset")

	by_location = split_by_location(data)
	chunks = [c for l in by_location for c in split_on_gaps(l, gap_detection)]
	timer = Timer()
	model = LSTMModel(vars_in=len(features_in), vars_out=len(features_out))

	# extract the data with a sliding window of length 20
	# x_train, y_train = prepare_window_off_by_1(data, 20)
	x_train = []
	y_train = []
	for chunk in chunks:
		if use_offset:
			x_train.append(slide_rows(chunk[features_in][:-1].to_numpy(), stride))
			y_train.append(slide_rows(chunk[features_out][1:].to_numpy(), stride))
		else:
			x_train.append(slide_rows(chunk[features_in].to_numpy(), stride))
			y_train.append(slide_rows(chunk[features_out].to_numpy(), stride))

	x_train = np.concatenate(x_train)
	y_train = np.concatenate(y_train)

	logger.info("Training LSTM...")
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

	model = train_lstm_model_predict(data)

	x_test, y_test = prepare_window_off_by_1(data[-50:], 20)
	y_pred = np.array([model.predict_next(x_test[i]) for i in range(x_test.shape[0])])  # model.predict(x_test, fh=10)
	y_pred_seq = model.predict_sequence(x_test[0], y_test.shape[0])

	plt.plot(y_test[:, -1, 0])  # blue
	plt.plot(y_pred[:, 0])  # green
	plt.plot(y_pred_seq[:, 0])  # orange
	plt.show()


if __name__ == '__main__':
	test_lstm_model()
