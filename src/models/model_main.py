from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
from sktime.datasets import load_airline, load_uschange, load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from utils.logger import Logger
from utils import sqlite_utils
import sqlalchemy
import pandas as pd
import numpy as np
from utils.timer import Timer

logger: Logger = Logger(module_name="Model 1")
script_timer = Timer()
# structure of time series: rows: instances, cols: variables, depth: series of values


def load_dataset(dataset: str, location: str) -> pd.DataFrame:
	"""
		Loads a dataset from the database. A dataset consists of exactly 1 timeseries taken from
		exactly 1 measuring station. It is represented by a table with a column for every variable
		and a row for every timepoint and returned in a pandas DataFrame.
		
		Parameters:
		
			dataset: The name of the dataset in the database. Currently this is either 'zurich' or 'nabel'
			
			location: The name of the measuring station. This is dependent on the selected dataset.
	"""
	logger.info_begin("Loading dataset...")
	timer = Timer()
	
	logger.info_end(f"Done in {timer}")


def main():



# if this is the main file, then run the main function directly
if __name__ == "__main__":
	main()

