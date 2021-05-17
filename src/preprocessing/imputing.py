from sklearn.impute import SimpleImputer

import pandas as pd

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Imputer")


def impute_simple_imputer(timeseries: pd.DataFrame):
	logger.info_begin("Imputing missing values...")
	timer = Timer()
	imputer = SimpleImputer(strategy='mean')
	imputed_timeseries = pd.DataFrame(imputer.fit_transform(timeseries), columns=timeseries.columns)
	logger.info_end(f"Done in {timer}")
	return imputed_timeseries
