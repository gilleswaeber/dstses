"""
	This module contains all code specifically necessary code for the LSTM model
"""
# disable tensorflow logging
import json
import os
from pathlib import Path
from typing import List, Tuple

from pandas import DataFrame
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from preprocessing.sliding_window import prepare_window_off_by_1, slide_rows
from utils.keras import KerasTrainCallback
from utils.normalizer import Normalizer

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
from configparser import SectionProxy

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
			self.model.add(LSTM(units=64, activation='sigmoid', input_shape=(None, vars_in), return_sequences=True))
			self.model.add(TimeDistributed(Dense(vars_out, activation="sigmoid")))
			# self.model.compile(loss='mean_squared_error')
			self.model.compile(optimizer=Adam(clipnorm=1), loss='mean_squared_error')
			self.model.summary()
		else:
			self.model = model

	def fit(self, /, x: np.ndarray, y: np.ndarray, epochs: int):

		print(f"Shape: {x.shape}")
		print(f"Model: {self.model.input_shape}")
		with tqdm(total=epochs, desc='LSTM training', dynamic_ncols=True) as progress:
			nntc = KerasTrainCallback(progress)
			self.model.fit(x=x, y=y, batch_size=32, validation_split=1/8, use_multiprocessing=True, callbacks=[nntc], epochs=epochs, verbose=0)
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

	def store(self, config: SectionProxy):
		path = Path(config["storage_location"]) / f"{config.name}.pkl"
		if not os.path.exists(path):
			self.model.save(path)

	def predict(self, x, fh):
		return self.predict_next(x)


def train_or_load_LSTM(config: SectionProxy, data: pd.DataFrame) -> Tuple[LSTMModel, SectionProxy]:
	path = Path(config["storage_location"]) / f"{config.name}.pkl"
	if path.exists() and False:
		return load(config, path), config
	else:
		return train_lstm_model_predict(config, data), config


def load(config: SectionProxy, path: Path) -> LSTMModel:
	return LSTMModel(vars_in=0, vars_out=0, model=load_model(path))


def split_by_location(df: DataFrame) -> List[DataFrame]:
	"""Split into several dataframes using the the part before the dot in the column name as criterion"""
	df_cols = list(df.columns)
	locations = set(c.split('.')[0] for c in df_cols)
	by_location = []
	for l in locations:
		loc_cols = [c for c in df_cols if c.split('.')[0] == l]
		by_location.append(df[loc_cols].rename(columns={c: c.split('.', 1)[1] for c in loc_cols}))
	return by_location


def split_on_gaps(df: DataFrame, gap_gte_seconds) -> List[DataFrame]:
	"""Drop rows with NaN"""
	chunks = []
	df = df.dropna()  # Drop NaNs
	gaps = ((df.index[1:] - df.index[:-1]).seconds >= gap_gte_seconds).nonzero()[0].tolist()
	gaps.append(len(df))
	start = 0
	for gap in gaps:
		chunks.append(df[start:gap])
		start = gap
	return chunks


class LSTMConfig:
	def __init__(self, config: SectionProxy):
		self.gap_detection = int(config["gap_detection_seconds"])
		self.stride_length = int(config["train_window"])
		self.features_in: List[str] = json.loads(config["features_in"])
		self.features_out: List[str] = json.loads(config["features_out"])
		self.use_offset: bool = config.getboolean("use_offset")


def to_x_y(df: pd.DataFrame, c: LSTMConfig) -> Tuple[np.ndarray, np.ndarray]:
	by_location = split_by_location(df)
	chunks = [chunk for l in by_location for chunk in split_on_gaps(l, c.gap_detection) if chunk.shape[0] > c.stride_length]

	# extract the data with a sliding window of length 20
	x = []
	y = []
	for chunk in chunks:
		if c.use_offset:
			x.append(slide_rows(chunk[c.features_in][:-1].to_numpy(), c.stride_length))
			y.append(slide_rows(chunk[c.features_out][1:].to_numpy(), c.stride_length))
		else:
			x.append(slide_rows(chunk[c.features_in].to_numpy(), c.stride_length))
			y.append(slide_rows(chunk[c.features_out].to_numpy(), c.stride_length))

	x = np.concatenate(x)
	y = np.concatenate(y)

	normalizer = Normalizer()
	for col, f in enumerate(c.features_in):
		x[:, :, col] = normalizer.normalize(x[:, :, col], f)
	for col, f in enumerate(c.features_out):
		y[:, :, col] = normalizer.normalize(y[:, :, col], f)

	return x, y


def train_lstm_model_predict(config: SectionProxy, df: pd.DataFrame) -> LSTMModel:
	"""Several things to do here: split by location, do a gap detection to split in chunks, pass each chunk through the sliding window function, …"""
	logger.info("Preparing LSTM training data...")
	c = LSTMConfig(config)

	x_train, y_train = to_x_y(df, c)

	model = LSTMModel(vars_in=len(c.features_in), vars_out=len(c.features_out))

	timer = Timer()
	logger.info("Training LSTM...")
	model.fit(x=x_train, y=y_train, epochs=10)
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
