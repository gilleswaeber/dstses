from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import mape_loss

from preprocessing.imputing import impute_simple_imputer
from models.scoring import eval_model_mape

from utils.logger import Logger
from utils.timer import Timer

import numpy as np
import pandas as pd

logger = Logger("Experiment")
timer_script = Timer()


def get_labels(labels: list, filters: list) -> list:
	out = []
	for f in filters:
		for lbl in labels:
			if f in lbl:
				out.append(lbl)
	
	return list(set(out))


def load_dataset(timeseries: pd.DataFrame) -> (pd.Series, pd.DataFrame):
	logger.info_begin("Loading dataset...")
	timer = Timer()

	# select only intervals where all values are available
	fvi = np.max([timeseries[col].first_valid_index() for col in timeseries.columns])
	drop_indices = np.arange(0, fvi+1)
	timeseries = timeseries.drop(drop_indices)
	# drop the date column because it is not a numeric value
	timeseries = timeseries.drop(labels=["date"], axis=1)
	logger.info_end(f'Done in {timer}')
	
	return timeseries


def transform_data(timeseries: pd.DataFrame) -> (pd.Series, pd.DataFrame):
	logger.info_begin("Transforming data...")
	timer = Timer()
	# create x and y from the dataset (exclude date and y from x)
	y = timeseries[filter(lambda v: "PM10" in v, timeseries.columns)].squeeze()
	x = timeseries.drop(labels=get_labels(timeseries.columns, ["PM10", "PM2.5", "PM1"]), axis=1, errors='ignore')
	logger.info_end(f'Done in {timer}')
	return y, x


def train_model_autoarima(y, x) -> AutoARIMA:
	logger.info_begin("Training AutoARIMA model...")
	timer = Timer()
	model = AutoARIMA()
	
	model.fit(y, x)
	
	logger.info_end(f'Done in {timer}')
	return model


def prepare_model(timeseries: pd.DataFrame):
	logger.info("Running script...")

	timeseries = load_dataset(timeseries)
	imputed_timeseries = impute_simple_imputer(timeseries)
	y, x = transform_data(imputed_timeseries)
	y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.1)
	model = train_model_autoarima(y_train, x_train)
	score = eval_model_mape(model, y_test, x_test)
	logger.info(f"Score of model: {score:.04f}")
	logger.info(f"Completed script in {timer_script}")