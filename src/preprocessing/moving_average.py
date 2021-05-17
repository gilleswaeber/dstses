"""

"""

import pandas as pd
import numpy as np

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Moving Average")


def moving_average(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""
		Computes a moving average over a timeseries.
	"""
	logger.info_begin("Computing moving average...")
	timer = Timer()
	
	tau = 3.0
	window = 5.0
	# correction factor for keeping the scale of the data consistent
	a = np.e ** (-window / tau)
	f = (a*(np.e ** ((1.0 / tau)*(window + 1))) - a) / (np.e ** (1.0 / tau) - 1.0) * np.sqrt(2)
	dataframe = dataframe.rolling(int(window), win_type='exponential', center=True).sum(tau=tau, sym=True).div(f)
	
	logger.info_end(f"Done in {timer}")
	
	return dataframe