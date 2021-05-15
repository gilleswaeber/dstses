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


def load_dataset(dataset: str, location: str) -> pd.DataFrame:
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
	
	# get the db engine and connect to the database
	engine = sqlite_utils.get_engine().connect()
	
	# read the dataframe from the database
	df: pd.DataFrame = sqlite_utils.get_time_series(engine=engine, dataset=dataset, location=location)
	
	logger.info_end(f"Done in {timer}")
	return df

def main():
	print_header()
	timer_main = Timer()
	
	# define some variables for use in this function
	dataset = 'zurich'
	location = 'Zch_Stampfenbachstrasse'
	measurements_input = ['Humidity', 'Temperature', 'Pressure']
	measurements_output = ['PM10']
	col_names_timestamp = ['date']
	col_names_input = [f'{location}.{v_name}' for v_name in measurements_input]
	col_names_output = [f'{location}.{v_name}' for v_name in measurements_output]
	
	# read and prepare dataset for training
	df_timeseries_complete, headers = load_dataset(dataset, location)
	df_timestamps, df_input, df_output = select_columns_3(df_timeseries_complete[-250:], col_names_timestamp, col_names_input, col_names_output)
	
	# testing...
	plt.plot(df_input)
	plt.plot(df_output)
	plt.show()
	
	logger.info("Script completed in {timer}.")
	logger.info("Terminating gracefully...")


# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main()

