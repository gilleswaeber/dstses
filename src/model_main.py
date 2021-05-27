import asyncio
from asyncio import gather
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split

from adapters.data_adapter import IDataAdapter
from adapters.hist_data_adapter import HistDataAdapter
from adapters.influx_sensordata import InfluxSensorData
from models.autoarima import train_or_load_ARIMA
from models.lstm import train_or_load_LSTM
from models.modelholder import ModelHolder
from preprocessing.column_selector import select_columns_3
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
	logger.info_begin("Loading dataset...")
	timer = Timer()

	adapter: IDataAdapter = HistDataAdapter(config, name)
	# read the dataframe from the database
	df: pd.DataFrame = adapter.get_data()

	logger.info_end(f"Done in {timer}")
	return df


def chop_first_fringe(timeseries: pd.DataFrame) -> pd.DataFrame:
	logger.info("Loading dataset...")
	timer = Timer()

	# select only intervals where all values are available
	fvi = np.max([timeseries[col].first_valid_index() for col in timeseries.columns])
	timeseries = timeseries[fvi:].reset_index(drop=True)
	timeseries.set_index('date', inplace=True)
	logger.info(f'Done in {timer}')

	return timeseries

def main_executor():
	asyncio.run(main())

async def main():
	print_header()
	timer_main = Timer()

	config = default_config()

	# read and prepare dataset for training
	df_timeseries_complete = load_dataset("zurich_adapter", config)

	print(df_timeseries_complete[:1])
	df_timeseries = chop_first_fringe(df_timeseries_complete)
	df_train_val, df_test = temporal_train_test_split(df_timeseries, test_size=.20)

	models = [
		# ModelHolder(name="arima", trainer=train_or_load_ARIMA, config=config),
		ModelHolder(name="lstm", trainer=train_or_load_LSTM, config=config)
	]

	trainers = [to_thread(model.trainer, config=model.config, data=df_train_val) for model in models]
	models = await gather(*trainers)
	[model.store(conf) for model, conf in models]  # Stores if not existing. Does NOT OVERWRITE!!!

	forecast_test = [model.predict(x=df_test, fh=5) for model in models]

	print(forecast_test)

	plt.plot(forecast_test)
	plt.show()

	logger.info(f"Script completed in {timer_main}.")
	logger.info("Terminating gracefully...")

	logger.info("start predicting new time")

	with InfluxSensorData(config=config, name="influx") as client:
		data = client.get_data()
		imputed_data = impute_simple_imputer(data)
		avg_data = moving_average(imputed_data)
		forecast_list = [model.predict(x=avg_data, fh=5) for model in models]
		forecast=sum(forecast_list)/len(forecast_list)
		client.send_data(forecast)


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main_executor()
