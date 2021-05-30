import asyncio
import json
from asyncio import gather
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split

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

	# read and prepare dataset for training
	df_timeseries_complete = load_dataset("zurich_adapter", config)

	df_timeseries = chop_first_fringe(df_timeseries_complete) # Chop first improper filled rows
	imputed_timeseries = impute_simple_imputer(df_timeseries)
	smooth_timeseries = moving_average(imputed_timeseries)
	smooth_timeseries.dropna(inplace=True) # Make sure there really is no empty cell anymore, else drop row
	# Split training/testing data in 80%/20%
	df_train_val, df_test = temporal_train_test_split(smooth_timeseries, test_size=.20)

	# Define all models at our disposal
	models = [
		ModelHolder(name="arima", trainer=train_or_load_ARIMA, config=config),
		ModelHolder(name="autoarima", trainer=train_or_load_autoARIMA, config=config),
		ModelHolder(name="expsmooting", trainer=train_or_load_expSmoothing, config=config),
		ModelHolder(name="lstm", trainer=train_or_load_LSTM, config=config),
		ModelHolder(name="lstm_seq", trainer=train_or_load_LSTM, config=config)
	]

	# Train the models
	trained_models = await gather(*[to_thread(train_model, model=model, data=df_train_val) for model in models])
	[model.model.store(model.config) for model in trained_models]  # Stores if not existing. Does NOT OVERWRITE!!!

	# Test the generalization performance of our models
	forecast_test = [model.model.predict(x=df_test, fh=5) for model in trained_models]

	print(forecast_test)

	# plt.plot(forecast_test[0][['Zch_Stampfenbachstrasse.PM10', 'Zch_Stampfenbachstrasse.PM10_Pred']])
	# plt.plot(forecast_test[0][['Zch_Stampfenbachstrasse.Humidity', 'Zch_Stampfenbachstrasse.Temperature']])
	# plt.show()

	logger.info(f"Script completed in {timer_main}.")
	logger.info("Terminating gracefully...")

	logger.info("start predicting new time")

	with InfluxSensorData(config=config, name="influx") as client:
		# Load the data from the server
		data = client.get_data()
		imputed_data = impute_simple_imputer(data) # Impute
		avg_data = moving_average(imputed_data) # Average input
		logger.debug("Forecasting")
		forecast_list = [model.model.predict(x=avg_data, fh=5) for model in trained_models] # Make predictions

		logger.info(forecast_list)
		forecast_dict = {
			"arima": forecast_list[0],
			"autoarima": forecast_list[1],
			"expsmoothing": forecast_list[2],
			"lstm": forecast_list[3].iloc[:, forecast_list[3].columns.get_loc("Live.PM10_Pred")],
			"lstm_seq": forecast_list[4].iloc[:, forecast_list[4].columns.get_loc("Live.PM10_Pred")]
		}

		forecast = pd.DataFrame(data=forecast_dict)
		logger.debug(forecast)
		forecast=forecast.mean(axis=1).head(n=5)
		logger.info(f"Forcasting finished with forecast value\n {forecast}")


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main_executor()
