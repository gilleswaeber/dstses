"""
	This file contains some functions that graph datasets, either loaded or generated.
"""
import os

from utils.config import default_config

from adapters.influx_sensordata import InfluxSensorData
from adapters.hist_data_adapter import HistDataAdapter

import matplotlib.pyplot as plt
import numpy as np

from utils.sqlite_utils import get_engine, get_time_series

# attempt to use the config framework
config = default_config()


def graph_random_noise():
	"""
		Graphs 100 steps of a randomly generated timeseries
	"""
	data = np.random.random(100)

	fig, ax = plt.subplots(1)
	ax.set_title("Random Noise")
	ax.plot(data)
	fig.savefig("random_noise.png")


def graph_comparison_our_vs_hist():
	"""
		Graphs a day of our data and the same day of historical data
	"""
	adapter_hist = HistDataAdapter(config, "graph_comparison_our_hist_hist")
	adapter_influx = InfluxSensorData(config, "graph_comparison_our_hist_influx")
	
	hist_data = adapter_hist.get_data()
	our_data = adapter_influx.get_data()
	
	print()
	print(type(hist_data))
	print(type(our_data))
