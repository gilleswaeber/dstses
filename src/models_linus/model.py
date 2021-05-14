from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
from sktime.datasets import load_airline, load_uschange, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from opendata_converter.logger import Logger
from models_linus import sqlite_utils
import sqlalchemy
import pandas as pd
import numpy as np
from models_linus.timer import Timer

logger: Logger = Logger(module_name="Model 1")
# structure of time series: rows: instances, cols: variables, depth: series of values


def load_timeseries(engine: sqlalchemy.engine.Connection, timeseries: list) -> pd.DataFrame:
	logger.info_begin("Loading timeseries...")
	timer = Timer()
	i = 1
	loaded_timeseries: list = []
	for dataset, location in timeseries:
		logger.info_update(f'{i}/{len(timeseries)}')
		df: pd.DataFrame = sqlite_utils.get_time_series(engine, dataset, location)
		# TODO: translate dataframe format
		loaded_timeseries.append(df)
	logger.info_end(f'Done in {timer}.')
	return loaded_timeseries


def main():
	logger.info_begin("Connection to database...")
	engine = sqlite_utils.get_engine().connect()
	logger.info_end("Done.")
	
	timeseries_to_load = [('zurich', 'Zch_Stampfenbachstrasse'), ('zurich', 'Zch_Schimmelstrasse')]
	timeseries: list = load_timeseries(engine, timeseries_to_load)
	
	print(timeseries)
	# y_train, y_test = temporal_train_test_split(timeseries.squeeze(), fh=168)
	# print(y_train, y_test)


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main()

