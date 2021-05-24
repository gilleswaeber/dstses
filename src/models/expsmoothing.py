import numpy as np
import pandas as pd
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import temporal_train_test_split

from models.scoring import eval_model_mape
from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average
from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Exp Smoothing")
timer_script = Timer()


def get_labels(labels: list, filters: list) -> list:
	out = []
	for f in filters:
		for lbl in labels:
			if f in lbl:
				out.append(lbl)

	return list(set(out))


def load_dataset(timeseries: pd.DataFrame, output: bool = True) -> (pd.Series, pd.DataFrame):
	if output:
		logger.info_begin("Loading dataset...")
		timer = Timer()

	# select only intervals where all values are available
	fvi = np.max([timeseries[col].first_valid_index() for col in timeseries.columns])
	drop_indices = np.arange(0, fvi + 1)
	timeseries = timeseries.drop(drop_indices)
	# drop the date column because it is not a numeric value
	timeseries = timeseries.drop(labels=["date"], axis=1)
	if output:
		logger.info_end(f'Done in {timer}')

	return timeseries


def transform_data(timeseries: pd.DataFrame, output: bool = True) -> (pd.Series, pd.DataFrame):
	if output:
		logger.info_begin("Transforming data...")
		timer = Timer()
	# create x and y from the dataset (exclude date and y from x)
	y = timeseries[filter(lambda v: "PM10" in v, timeseries.columns)].squeeze()
	x = timeseries.drop(labels=get_labels(timeseries.columns, ["PM10", "PM2.5", "PM1"]) + ["date"], axis=1,
						errors='ignore')

	if output:
		logger.info_end(f'Done in {timer}')
	return y, x


def train_model_expsmooth(y, x, output: bool = True) -> ExponentialSmoothing:
	if output:
		logger.info_begin("Training AutoARIMA model...")
		timer = Timer()
	model = ExponentialSmoothing()

	model.fit(y, x)

	if output:
		logger.info_end(f'Done in {timer}')
	return model


def prepare_model(timeseries: pd.DataFrame, output: bool = True):
	if output:
		logger.info("Running script...")

	timeseries = load_dataset(timeseries, output)
	imputed_timeseries = impute_simple_imputer(timeseries, output)
	smooth_timeseries = moving_average(imputed_timeseries, output)
	y, x = transform_data(smooth_timeseries, output)
	y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=0.1)
	model = train_model_expsmooth(y_train, x_train, output)
	score = eval_model_mape(model, y_test, x_test, output)
	if output:
		logger.info(f"Score of model: {score:.04f}")
		logger.info(f"Completed script in {timer_script}")
