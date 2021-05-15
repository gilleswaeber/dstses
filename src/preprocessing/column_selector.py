"""

"""

import pandas as pd
from typing import List

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Column Selector")


def select_columns_3(df_input: pd.DataFrame, time_cols: List[str], in_cols: List[str], out_cols: List[str])\
		-> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
	"""
		Splits a pandas DataFrame along its columns into three DataFrames containing timestamps, a DataFrame containing
		the input data and a DataFrame containing the output data.
		
		Parameters:
			
			df_input: The pandas DataFrame containing all the data of a given dataset.
			
			time_cols: The list of column names of all columns that contain timestamps (or similar), that should not be
			used to train upon but might be needed for labelling plots later on.
			
			in_cols: The list of all column names that contain the input data for the model.
			
			out_cols: The list of all column names that contain the output data for the model.
		
		Returns:
			
			Three pandas DataFrames for the timestamps, the input data and the output data in this order.
	"""
	logger.info_begin("Splitting columns...")
	timer = Timer()
	
	df_time = df_input[time_cols]
	df_in = df_input[in_cols]
	df_out = df_input[out_cols]
	
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
