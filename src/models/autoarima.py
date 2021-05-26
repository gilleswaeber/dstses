import pandas as pd
from configparser import ConfigParser

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


def transform_data(timeseries: pd.DataFrame, output: bool = True) -> (pd.Series, pd.DataFrame):
	if output:
		logger.info("Transforming data...")
		timer = Timer()
	# create x and y from the dataset (exclude date and y from x)
	y = timeseries[filter(lambda v: "PM10" in v, timeseries.columns)]
	y_columns = len(y.columns)
	x = timeseries.drop(labels=get_labels(timeseries.columns, ["PM10", "PM2.5", "PM1"]) + ["date"], axis=1,
						errors='ignore')
	x_series = pd.Series()
	for i in range(y_columns):
		x_series = x_series.append(x)

	if output:
		logger.info(f'Done in {timer}')
	return y.squeeze(axis=1), x_series


def train_model_autoarima(y, x, output: bool = True) -> ARIMA:
	if output:
		logger.info("Training AutoARIMA model...")
		timer = Timer()
	model = AutoARIMA(suppress_warnings=True, error_action='ignore')

	model.fit(y, x)

	if output:
		model.summary()
		logger.info(f'Done in {timer}')
	return model


class ArimaModel:
	def __init__(self, model : ARIMA = None):
		if model is None:
			self.model = ARIMA()
		else:
			self.model = model

	def prepare_model(self, timeseries: pd.DataFrame, output: bool = True) -> ARIMA:
		if output:
			logger.info("Running script...")

		imputed_timeseries = impute_simple_imputer(timeseries, output)
		smooth_timeseries = moving_average(imputed_timeseries, output)
		y, x = transform_data(smooth_timeseries, output)
		y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.1)
		model = train_model_autoarima(y_train, x_train, output)
		score = eval_model_mape(model, y_test, x_test, output)
		if output:
			logger.info(f"Score of model: {score:.04f}")
			logger.info(f"Completed script in {timer_script}")
		return model

	def predict(self, x, fh: int):
		return self.model.predict(X=x, fh=fh)

	def store(self, config: ConfigParser):
		path = config["storage_location"] + "/autoarima.pkl"
		if not os.path.exists(path):
			with open(path, 'wb') as pkl:
				pickle.dump(self.model, pkl)

def train_or_load_ARIMA(config: ConfigParser, data: pd.DataFrame) -> ArimaModel:
	p = config["storage_location"] + "/autoarima.pkl"
	if os.path.exists(p):
		return load(config)
	else:
		data = moving_average(data)
		model = ArimaModel()
		model.prepare_model(data)
		return model

def load(config: ConfigParser) -> ArimaModel:
	path = config["storage_location"] + "/autoarima.pkl"
	with open(path, "rb") as pkl:
		return ArimaModel(model=pickle.load(pkl))
