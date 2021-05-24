"""

"""

import json
from configparser import ConfigParser
from typing import List

import pandas as pd

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Column Selector")


def select_columns_3(df_input: pd.DataFrame, config: ConfigParser, name: str) -> (
pd.DataFrame, pd.DataFrame, pd.DataFrame):
	"""
		Splits a pandas DataFrame along its columns into three DataFrames containing timestamps, a DataFrame containing
		the input data and a DataFrame containing the output data.

		Uses following configuration parameters under [preprocessing]:
		measurements_timestamp: The list of column names of all columns that contain timestamps (or similar), that should not be
			used to train upon but might be needed for labelling plots later on.

		measurements_input: The list of all column names that contain the input data for the model.

		measurements_output: The list of all column names that contain the output data for the model.
		
		Parameters:
			
			df_input: The pandas DataFrame containing all the data of a given dataset.
			
			config: The global configuration object to load parameters from.

			name: Name of preprocessor in the configuration
		
		Returns:
			
			Three pandas DataFrames for the timestamps, the input data and the output data in this order.
	"""
	logger.info_begin("Splitting columns...")
	timer = Timer()

	location = config[name]["location"]
	measurements_input = json.loads(config[name]["measurements_input"])
	measurements_output = json.loads(config[name]["measurements_output"])
	col_names_timestamp = json.loads(config[name]["measurements_timestamp"])
	col_names_input = [f'{location}.{v_name}' for v_name in measurements_input]
	col_names_output = [f'{location}.{v_name}' for v_name in measurements_output]

	df_time = df_input[col_names_timestamp]
	df_in = df_input[col_names_input]
	df_out = df_input[col_names_output]

	logger.info_end(f"Done in {timer}")

	return df_time, df_in, df_out


def select_columns_2(df_input: pd.DataFrame, a_cols: List[str], b_cols: List[str]) -> (pd.DataFrame, pd.DataFrame):
	"""
		Splits a pandas DataFrame along its columns into two DataFrames each one containing a specified set of
		columns of the input.
		
		Parameters:
			
			df_input: The input DataFrame containing the entire dataset.
			
			a_cols: The list of column names that should be written in the left DataFrame.
			
			b_cols: The list of column names that should be written in the right DataFrame.
		
		Returns:
			
			Two pandas DataFrames containing the columns in a_cols and b_cols respectively.
	"""
	logger.info_begin("Splitting columns (2 groups)...")
	timer = Timer()

	df_a = df_input[a_cols]
	df_b = df_input[b_cols]

	logger.info_end(f"Done in {timer}")

	return df_a, df_b
