import pandas as pd
from sklearn.impute import SimpleImputer

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Imputer")


def impute_simple_imputer(timeseries: pd.DataFrame, output: bool = True):
	"""
	Imputes the missing values with a simple mean to fill it up.

	:param timeseries: Timeseries to fill.
	:param output:
	:return: A imputed timeseries
	"""
	if output:
		logger.info_begin("Imputing missing values...")
		timer = Timer()
	imputer = SimpleImputer(strategy='mean')
	imputed_timeseries = pd.DataFrame(imputer.fit_transform(timeseries), columns=timeseries.columns,
									  index=timeseries.index)
	if output:
		logger.info_end(f"Done in {timer}")
	return imputed_timeseries
