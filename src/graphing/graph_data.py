"""
	This file contains some functions that graph datasets, either loaded or generated.
"""
import os

from configparser import ConfigParser
from pathlib import Path

from adapters.influx_sensordata import InfluxSensorData
from adapters.hist_data_adapter import HistDataAdapter

import matplotlib.pyplot as plt
import numpy as np

from utils.sqlite_utils import get_engine, get_time_series

# attempt to use the config framework
CONFIG_FILE = Path(os.getenv("FILE_PATH", __file__)).parent.parent.absolute() / "resources" / "config.ini"
config = ConfigParser()
config.read(str(CONFIG_FILE))
config['DEFAULT']['resources_path'] = str(Path(CONFIG_FILE).parent.absolute())


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
