"""

"""

import pandas as pd
import numpy as np

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Moving Average")


def moving_average(dataframe: pd.DataFrame, output: bool = True) -> pd.DataFrame:
	"""
		Computes a moving average over a timeseries.
	"""
	if output:
		logger.info_begin("Computing moving average...")
		timer = Timer()
	
	tau = 3.0
	window = 5.0
	
	# correction factor for keeping the scale of the data consistent
	a = np.e ** (-window / tau)
	f = (a*(np.e ** ((1.0 / tau)*(window + 1))) - a) / (np.e ** (1.0 / tau) - 1.0)
	
	# compute residual error of correction factor, because thanks pandas
	tmp = pd.DataFrame(np.ones(100))
	corr_factor = tmp.rolling(int(window), win_type='exponential', center=True).sum(tau=tau, sym=True).div(f).mean()[0]
	
	dataframe = pd.concat([dataframe[:1], dataframe[:1], dataframe, dataframe[-1:], dataframe[-1:]]).rolling(int(window), win_type='exponential', center=True).sum(tau=tau, sym=True).div(f * corr_factor)
	
	if output:
		logger.info_end(f"Done in {timer}")
	
	return dataframe[2:-2]
