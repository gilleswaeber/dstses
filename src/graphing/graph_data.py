"""
	This file contains some functions that graph datasets, either loaded or generated.
"""
import os
import sys

from utils.config import default_config

from adapters.influx_sensordata import InfluxSensorData
from adapters.hist_data_adapter import HistDataAdapter

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.sqlite_utils import get_engine, get_time_series


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
	_from = "2021-05-10 00:00:00"
	_to = "2021-05-15 00:00:00"
	
	config = default_config()
	
	adapter_hist = HistDataAdapter(config, "graph_comparison_our_hist_hist")
	adapter_influx = InfluxSensorData(config, "graph_comparison_our_hist_influx")
	
	hist_data = adapter_hist.get_data(output=False)
	our_data = adapter_influx.get_data()
	
	our_data[0].drop(labels=["table", "_start", "_stop", "device", "sensor", "result", "_measurement"], axis=1, inplace=True)
	our_data[1].drop(labels=["table", "_start", "_stop", "device", "result", "_measurement"], axis=1, inplace=True)
	our_data[2].drop(labels=["table", "_start", "_stop", "device", "sensor", "result", "_measurement"], axis=1, inplace=True)
	
	# filter the right fields where necessary
	our_pm10 = our_data[1][our_data[1]["_field"] == "pm10"]
	our_humidity = our_data[0]
	our_temperature = our_data[2]
	
	# filter the right place (for now 8610)
	our_pm10 = our_pm10[our_pm10["place"] == "8610"]
	our_humidity = our_humidity[our_humidity["place"] == "8610"]
	our_temperature = our_temperature[our_temperature["place"] == "8610"]
	
	our_pm10 = our_pm10.drop(labels=["_field", "place"], axis=1)
	our_humidity = our_humidity.drop(labels=["_field", "place"], axis=1)
	our_temperature = our_temperature.drop(labels=["_field", "place"], axis=1)
	
	our_pm10.columns = ["date", "PM10"]
	our_humidity.columns = ["date", "Humidity"]
	our_temperature.columns = ["date", "Temperature"]
	
	our_pm10.reset_index(inplace=True)
	our_humidity.reset_index(inplace=True)
	our_temperature.reset_index(inplace=True)
	
	# join the columns together
	our_data = our_pm10.join(our_humidity.join(our_temperature, rsuffix="_r1"), rsuffix="_r0")
	our_data.drop(labels=["index", "index_r0", "index_r1", "date_r0", "date_r1"], axis=1, inplace=True)
	our_data.set_index("date", inplace=True)
	
	# select some days
	our_data = our_data[our_data.index.to_series() < _to]
	our_data = our_data[our_data.index.to_series() >= _from]
	
	# prepare historical data
	hist_data.set_index("date", inplace=True)
	hist_data = hist_data[hist_data.index.to_series() < _to]
	hist_data = hist_data[hist_data.index.to_series() >= _from]
	
	print(hist_data.columns)
	
	# plot our data
	fig, ax = plt.subplots(1)
	ax.set_title("Our (solid) vs Historical (dashed) Data")
	ax.plot(our_data["Humidity"], color="blue")
	ax.plot(our_data["Temperature"], color="red")
	ax.plot(our_data["PM10"], color="grey")
	
	ax.plot(hist_data["Zch_Stampfenbachstrasse.Humidity"], color="blue", linestyle="dashed")
	ax.plot(hist_data["Zch_Stampfenbachstrasse.Temperature"], color="red", linestyle="dashed")
	ax.plot(hist_data["Zch_Stampfenbachstrasse.PM10"], color="grey", linestyle="dashed")
	
	fig.tight_layout()
	fig.savefig("comparison_out_hist.png")


def get_nth_low(lst, n) -> float:
	lst.sort()
	length = len(lst)
	i = int(length * ((100.0 - n) / 100.0))
	return lst[i]


def get_nth_high(lst, n) -> float:
	lst.sort()
	length = len(lst)
	i = int(length * (n / 100.0))
	return lst[i]


def plot_high_low(high: List[float], low: List[float], title: str, name: str, ylim: List[float] = None):
	fig, ax = plt.subplots(1)
	ax.set_title(title)
	
	ax.fill_between(np.linspace(0, 23, 24), y1=low, y2=high)
	if ylim is not None:
		ax.set_ylim(ylim)
	
	fig.savefig(name)


def graph_typical_day():
	"""
		Graphs the typical day down to a 90th percentile
	"""
	
	_from = "2020-05-04 00:00:00"
	_to = "2021-05-03 00:00:00"
	
	config = default_config()
	
	adapter_hist = HistDataAdapter(config, "graph_typical_day")
	
	hist_data = adapter_hist.get_data(output=False).drop(labels=["Zch_Stampfenbachstrasse.PM2.5"], axis=1)
	hist_data.set_index("date", inplace=True)
	hist_data = hist_data[hist_data.index.to_series() <= _to]
	hist_data = hist_data[hist_data.index.to_series() >= _from]
	
	data_pm10 = {i: [] for i in range(24)}
	data_humidity = {i: [] for i in range(24)}
	data_temperature = {i: [] for i in range(24)}
	data_pressure = {i: [] for i in range(24)}
	
	for line in hist_data.to_records():
		hour = pd.Timestamp(line["date"]).hour
		hum = line["Zch_Stampfenbachstrasse.Humidity"]
		temp = line["Zch_Stampfenbachstrasse.Temperature"]
		pres = line["Zch_Stampfenbachstrasse.Pressure"]
		pm10 = line["Zch_Stampfenbachstrasse.PM10"]
		
		if not np.isnan(pm10):
			data_pm10[hour].append(pm10)
		if not np.isnan(hum):
			data_humidity[hour].append(hum)
		if not np.isnan(temp):
			data_temperature[hour].append(temp)
		if not np.isnan(pres):
			data_pressure[hour].append(pres)
			
	n = 90
	data_pm10_low = [get_nth_low(lst, n) for lst in data_pm10.values()]
	data_pm10_high = [get_nth_high(lst, n) for lst in data_pm10.values()]
	data_humidity_low = [get_nth_low(lst, n) for lst in data_humidity.values()]
	data_humidity_high = [get_nth_high(lst, n) for lst in data_humidity.values()]
	data_pressure_low = [get_nth_low(lst, n) for lst in data_pressure.values()]
	data_pressure_high = [get_nth_high(lst, n) for lst in data_pressure.values()]
	data_temperature_low = [get_nth_low(lst, n) for lst in data_temperature.values()]
	data_temperature_high = [get_nth_high(lst, n) for lst in data_temperature.values()]
	
	plot_high_low(data_pm10_low, data_pm10_high, "PM10", "typical_pm10.png")
	plot_high_low(data_humidity_low, data_humidity_high, "Humidity", "typical_humidity.png", [0, 100])
	plot_high_low(data_pressure_low, data_pressure_high, "Pressure", "typical_pressure.png", [900, 1000])
	plot_high_low(data_temperature_low, data_temperature_high, "Temperature", "typical_temperature.png", [-5, 30])

