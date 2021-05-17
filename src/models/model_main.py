from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
from sktime.datasets import load_airline, load_uschange, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
import sqlalchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import sqlite_utils
from utils.logger import Logger
from utils.timer import Timer
from preprocessing.column_selector import select_columns_2, select_columns_3
from preprocessing.moving_average import moving_average
from configparser import ConfigParser
from pathlib import Path
from adapters.data_adapter import IDataAdapter
from adapters.hist_data_adapter import HistDataAdapter
import os

logger: Logger = Logger(module_name="Model 1")
script_timer = Timer()
CONFIG_FILE = Path(os.getenv("FILE_PATH", __file__)).parent.parent.absolute() / "resources" / "config.ini"
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

	adapter : IDataAdapter = HistDataAdapter(config, name)
	# read the dataframe from the database
	df: pd.DataFrame = adapter.get_data()
	
	logger.info_end(f"Done in {timer}")
	return df

def main():
	print_header()
	timer_main = Timer()

	config = ConfigParser()
	config.read(str(CONFIG_FILE))
	config['DEFAULT']['resources_path'] = str(Path(CONFIG_FILE).parent.absolute())
	
	# read and prepare dataset for training
	df_timeseries_complete = load_dataset("test_adapter", config)
	df_timestamps, df_input, df_output = select_columns_3(df_timeseries_complete[-250:], config, "preprocessing")
	
	# testing...
	plt.plot(df_input)
	plt.plot(df_output)
	plt.show()
	
	logger.info(f"Script completed in {timer_main}.")
	logger.info("Terminating gracefully...")


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main()

