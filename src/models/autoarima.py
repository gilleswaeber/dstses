from typing import Tuple, List

import pandas as pd
import numpy as np
from configparser import ConfigParser, SectionProxy

from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.model_selection import temporal_train_test_split

from models.scoring import eval_model_mape
from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average
from utils.logger import Logger
from utils.timer import Timer
import pickle
import os

logger = Logger("Experiment")
timer_script = Timer()


def get_labels(labels: list, filters: list) -> list:
	out = []
	for f in filters:
		for lbl in labels:
			if f in lbl:
				out.append(lbl)

	return list(set(out))

def split_by_location(df: pd.DataFrame) -> List[pd.DataFrame]:
	"""Split into several dataframes using the the part before the dot in the column name as criterion"""
	df_cols = list(df.columns)
	locations = set(c.split('.')[0] for c in df_cols)
	by_location = []
	for l in locations:
		loc_cols = [c for c in df_cols if c.split('.')[0] == l]
		by_location.append(df[loc_cols].rename(columns={c: c.split('.', 1)[1] for c in loc_cols}))
	return by_location

def transform_data(timeseries: pd.DataFrame, output: bool = True) -> Tuple[np.ndarray, np.ndarray]:
	if output:
		logger.info("Transforming data...")
		timer = Timer()
	# create x and y from the dataset (exclude date and y from x)
	features = set(c.split('.', 1)[1] for c in timeseries.columns)
	features_in = [c for c in features if 'PM' not in c]
	features_out = list(features.difference(features_in))
	x = []
	y = []
	for df in split_by_location(timeseries):
		x.append(df[features_in].to_numpy())
		y.append(df[features_out].to_numpy())
	x = np.concatenate(x)
	y = np.concatenate(y)

	if output:
		logger.info(f'Done in {timer}')
	return y, x


def train_model_autoarima(y, x, output: bool = True) -> AutoARIMA:
	if output:
		logger.info("Training AutoARIMA model...")
		timer = Timer()
	model = AutoARIMA(suppress_warnings=True, error_action='ignore')

	y = pd.Series(data=np.delete(y, 0))
	x = pd.DataFrame(data=x[:-1])

	model.fit(y, x)

	if output:
		model.summary()
		logger.info(f'Done in {timer}')
	return model


class ArimaModel:
	def __init__(self, conf: ConfigParser, model: AutoARIMA = None):
		if model is not None:
			self.model = model
		self.config = conf

	def prepare_model(self, timeseries: pd.DataFrame, output: bool = True) -> AutoARIMA:
		if output:
			logger.info("Running script...")

		imputed_timeseries = impute_simple_imputer(timeseries, output)
		smooth_timeseries = moving_average(imputed_timeseries, output)
		y, x = transform_data(smooth_timeseries, output)
		y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.01)
		self.model = train_model_autoarima(y_test, x_test, output)
		y_test = pd.Series(data=np.delete(y_test, 0))
		x_test = pd.DataFrame(data=x_test[:-1])
		score = eval_model_mape(self.model, y_test, x_test, output)
		if output:
			logger.info(f"Score of model: {score:.04f}")
			logger.info(f"Completed script in {timer_script}")
		return self.model

	def predict(self, x, fh: int):
		_, data = transform_data(x, True)
		fh = len(data)
		prediction = self.model.predict(X=data, fh=np.linspace(1, fh, fh))
		prediction.index = range(len(prediction))
		return prediction

	def store(self, config: SectionProxy):
		path = config["storage_location"] + "/autoarima.pkl"
		if not os.path.exists(path):
			with open(path, 'wb') as pkl:
				pickle.dump(self.model, pkl)


def train_or_load_ARIMA(name: str, config: ConfigParser, data: pd.DataFrame) -> ArimaModel:
	p = config["storage_location"] + "/autoarima.pkl"
	if os.path.exists(p):
		return load(config)
	else:
		data = moving_average(data)
		model = ArimaModel(config)
		model.prepare_model(data)
		return model


def load(config: ConfigParser) -> ArimaModel:
	path = config["storage_location"] + "/autoarima.pkl"
	with open(path, "rb") as pkl:
		return ArimaModel(config, model=pickle.load(pkl))
