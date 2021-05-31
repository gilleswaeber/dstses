import asyncio
from asyncio import gather
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
import seaborn as sns

from adapters.data_adapter import IDataAdapter
from adapters.hist_data_adapter import HistDataAdapter
from adapters.influx_sensordata import InfluxSensorData
from models.arima import train_or_load_ARIMA
from models.autoarima import train_or_load_autoARIMA
from models.expsmoothing import train_or_load_expSmoothing
from models.lstm import train_or_load_LSTM
from models.modelholder import ModelHolder
from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average
from utils.config import default_config
from utils.logger import Logger
from utils.threading import to_thread
from utils.timer import Timer

logger: Logger = Logger(module_name="Model 1")
script_timer = Timer()


# structure of time series: rows: instances, cols: variables, depth: series of values


def print_header() -> None:
	"""
		Prints a header for a more pleasant console user experience. Not that this is at all important,
		but i felt like it. Don't try to change my mind.
	"""
	logger.info("##############################################################")
	logger.info("#       Datascience in Techno-Socio-Economic Systems         #")
	logger.info("#                                                            #")
	logger.info("#                 Forecasting the PM10 value                 #")
	logger.info("##############################################################")
	return


def load_dataset(name: str, config: ConfigParser) -> pd.DataFrame:
	"""
		Loads a dataset from the database. A dataset consists of exactly 1 timeseries taken from
		exactly 1 measuring station. It is represented by a table with a column for every variable
		and a row for every timepoint and returned in a pandas DataFrame.
		
		TODO: (maybe) expand this to include adapters to influxdb
		
		Parameters:
		
			dataset: The name of the dataset in the database. Currently this is either 'zurich' or 'nabel'
			
			location: The name of the measuring station. This is dependent on the selected dataset.
		
		Returns:
			
			A pandas DataFrame containing all data that could be read from the database
	"""
	logger.info("Loading dataset...")
	timer = Timer()

	adapter: IDataAdapter = HistDataAdapter(config, name)
	# read the dataframe from the database
	df: pd.DataFrame = adapter.get_data()

	logger.info(f"Done in {timer}")
	return df


def chop_first_fringe(timeseries: pd.DataFrame) -> pd.DataFrame:
	"""
	Chops the first fringe of the training data s.t. the data inputed to the training will not have empty cells anymore

	:param timeseries: The timeseries to remove fringe
	:return: A timeseries with the first fully valued line onwards of the input
	"""

	logger.info("Loading dataset...")
	timer = Timer()

	# select only intervals where all values are available
	fvi = np.max([timeseries[col].first_valid_index() for col in timeseries.columns])
	timeseries = timeseries[fvi:].reset_index(drop=True)
	timeseries.set_index('date', inplace=True)
	logger.info(f'Done in {timer}')

	return timeseries

def main_executor():
	"""
	If you do not use jupyter to call main call this function instead which first creates an executor to call the
	parallelized main function. On my computer I can speed up learning this way as ARIMA, EXP-Smooting etc only use one
	core for learning.
	:return: Nothing
	"""
	asyncio.run(main())

def train_model(model: ModelHolder, data: pd.DataFrame):
	model.model = model.trainer(model.name, model.config, data)
	return model

async def main():
	"""
	Main function of the application.
	:return: Nothing.
	"""
	print_header()
	timer_main = Timer()

	config = default_config()

	logger.info("start predicting new time")

	config["influx"]["drops"] = '["pm1", "pm4.0", "result", "table", "_time"]'
	config["influx"]["limit"] = "10000"
	with InfluxSensorData(config=config, name="influx") as client:
		# Load the data from the server
		data = client.get_data()
		imputed_data = impute_simple_imputer(data) # Impute
		avg_data = moving_average(imputed_data) # Average input
		logger.info(f"data len {len(avg_data)}")

		sns.set_theme(style="darkgrid")
		g = sns.jointplot(x="pm2.5", y="pm10", data=avg_data, kind="reg", truncate=False, xlim=(0, 40), ylim=(0, 40),
																										   color="m", height=7)
		g2 = sns.jointplot(x="temperature", y="humidity", data=avg_data, kind="reg", truncate=False, color="m", height=7)
		g3 = sns.jointplot(x="humidity", y="pm10", data=avg_data, kind="reg", truncate=False, color="m", height=7)
		g4 = sns.jointplot(x="temperature", y="pm10", data=avg_data, kind="reg", truncate=False, color="m", height=7)


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main_executor()
