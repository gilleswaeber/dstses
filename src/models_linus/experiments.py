from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA
from sktime.performance_metrics.forecasting import mape_loss

from sklearn.impute import SimpleImputer

from opendata_converter.logger import Logger
from models_linus.timer import Timer
from models_linus.sqlite_utils import get_engine, get_time_series

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
	
	return out


def load_dataset(dataset: str, location: str) -> (pd.Series, pd.DataFrame):
	logger.info_begin("Loading dataset...")
	timer = Timer()
	engine = get_engine().connect()
	timeseries = get_time_series(engine, dataset, location)
	engine.close()
	
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


def impute_missing_values(timeseries: pd.DataFrame):
	imputer = SimpleImputer(strategy='mean')
	imputed_timeseries = pd.DataFrame(imputer.fit_transform(timeseries))
	return imputed_timeseries


def train_model_autoarima(y, x) -> AutoARIMA:
	logger.info_begin("Training AutoARIMA model...")
	timer = Timer()
	model = AutoARIMA()
	
	model.fit(y, x)
	
	logger.info_end(f'Done in {timer}')
	return model


def eval_model(model, y_test, x_test) -> float:
	logger.info_begin("Measuring performance metrics...")
	timer = Timer()
	logger.info_update("Computing")
	y_pred = model.predict(x_test)
	logger.info_update("Scoring")
	error = mape_loss(y_test, y_pred)
	logger.info_end(f'Done in {timer}')
	return error


def main():
	logger.info("Running script...")
	timeseries = load_dataset("zurich", "Zch_Stampfenbachstrasse")
	imputed_timeseries = impute_missing_values(timeseries)
	y, x = transform_data(imputed_timeseries)
	y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.1)
	model = train_model_autoarima(y_train, x_train)
	score = eval_model(model, y_test, x_test)
	logger.info(f"Score of model: {score:.04f}")
	logger.info(f"Completed script in {timer_script}")
	

if __name__ == '__main__':
	main()
