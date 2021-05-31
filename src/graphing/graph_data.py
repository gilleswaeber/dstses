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

import seaborn as sns

from utils.sqlite_utils import get_engine, get_time_series


def graph_random_noise():
	"""
		Graphs 100 steps of a randomly generated timeseries
	"""
	data = np.random.random(100)
	data1 = np.array(([np.nan] * 50) + np.random.random(50).tolist())
	
	data = np.array([data, data1]).transpose()
	
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(kind="line", data=data)
	plot.set_axis_labels("Index", "Value")
	plot.ax.set_title("White Noise")
	plot.tight_layout()
	plot.savefig("random_noise.png")


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
	our_data = adapter_influx.get_data_8610()
	
	our_data.columns = ["Date", "Humidity [%] (Our)", "PM10 [µg/m³] (Our)", "Temperature [°C] (Our)"]
	our_data.set_index("Date", inplace=True)
	
	# select some days
	our_data = our_data[our_data.index.to_series() < _to]
	our_data = our_data[our_data.index.to_series() >= _from]
	
	# prepare historical data
	hist_data.set_index("date", inplace=True)
	hist_data = hist_data[hist_data.index.to_series() < _to]
	hist_data = hist_data[hist_data.index.to_series() >= _from]
	
	hist_data.drop(labels=["Zch_Stampfenbachstrasse.Pressure", "Zch_Stampfenbachstrasse.PM2.5"], axis=1, inplace=True)
	hist_data.columns = ["PM10 [µg/m³] (Official)", "Humidity [%] (Official)", "Temperature [°C] (Official)"]
	our_data.index = our_data.index.tz_localize(None)
	
	data = our_data.join(hist_data, how="outer")
	data.sort_index(axis=1, inplace=True)
	
	# plot our data
	palette = sns.color_palette(["#2222ff", "#0000ff", "#777777", "#666666", "#ff2222", "#ff0000"])
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(kind="line", palette=palette, data=data, dashes=[(2, 2), "", (2, 2), "", (2, 2), ""])
	plot.tight_layout()
	plot.ax.set_title("Comparison Our Data vs Official Data")
	plot.fig.autofmt_xdate()
	plot.savefig("comparison_our_hist.png")
	
"""	fig, ax = plt.subplots(1)
	ax.set_title("Our (solid) vs Historical (dashed) Data")
	ax.plot(our_data["Humidity"], color="blue")
	ax.plot(our_data["Temperature"], color="red")
	ax.plot(our_data["PM10"], color="grey")
	
	ax.plot(hist_data["Zch_Stampfenbachstrasse.Humidity"], color="blue", linestyle="dashed")
	ax.plot(hist_data["Zch_Stampfenbachstrasse.Temperature"], color="red", linestyle="dashed")
	ax.plot(hist_data["Zch_Stampfenbachstrasse.PM10"], color="grey", linestyle="dashed")
	
	fig.tight_layout()
	fig.savefig("comparison_out_hist.png")"""


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
	
	for line in hist_data.to_records():
		hour = pd.Timestamp(line["date"]).hour
		hum = line["Zch_Stampfenbachstrasse.Humidity"]
		temp = line["Zch_Stampfenbachstrasse.Temperature"]
		pm10 = line["Zch_Stampfenbachstrasse.PM10"]
		
		if not np.isnan(pm10):
			data_pm10[hour].append(pm10)
		if not np.isnan(hum):
			data_humidity[hour].append(hum)
		if not np.isnan(temp):
			data_temperature[hour].append(temp)
	
	x_pm10 = []
	y_pm10 = []
	x_humidity = []
	y_humidity = []
	x_temperature = []
	y_temperature = []
	
	for x in range(24):
		for y in data_pm10[x]:
			x_pm10.append(x)
			y_pm10.append(y)
		for y in data_humidity[x]:
			x_humidity.append(x)
			y_humidity.append(y)
		for y in data_temperature[x]:
			x_temperature.append(x)
			y_temperature.append(y)
	
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(x=x_pm10, y=y_pm10, kind="line", ci="sd")
	plot.set(xlabel="Time of Day", ylabel="PM10 [µg/m³]", title="Typical Day of PM10")
	plot.tight_layout()
	plot.savefig("typical_pm10.png")

	sns.set_theme(style="darkgrid")
	plot = sns.relplot(x=x_humidity, y=y_humidity, kind="line", ci="sd")
	plot.set(xlabel="Time of Day", ylabel="Humidity [%]", title="Typical Day of Humidity")
	plot.tight_layout()
	plot.savefig("typical_humidity.png")

	sns.set_theme(style="darkgrid")
	plot = sns.relplot(x=x_temperature, y=y_temperature, kind="line", ci="sd")
	plot.set(xlabel="Time of Day", ylabel="Temperature [°C]", title="Typical Day of Temperature")
	plot.tight_layout()
	plot.savefig("typical_temperature.png")
