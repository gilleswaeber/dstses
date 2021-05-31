"""
	This file contains some functions that demonstrate the functionality of our preprocessing in
	a visual manner.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average
from utils.sqlite_utils import get_engine, get_time_series


def graph_moving_average():
	"""
		Graphs some part of out dataset once without moving average, once with moving average
	"""
	timeseries: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")
	timeseries = timeseries[["date", "Zch_Stampfenbachstrasse.Humidity"]][-750:-250]
	timeseries.set_index("date", inplace=True)
	data = impute_simple_imputer(timeseries, False)
	data_smooth = moving_average(data, False)
	
	data.columns = ["Humidity (Original)"]
	data_smooth.columns = ["Humidity (Smoothed)"]
	
	data_plt = data.join(data_smooth)
	
	sns.set(rc={"figure.figsize": (2, 1)})
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(data=data_plt, kind="line", dashes=["", ""], legend=False, aspect=2)
	plt.setp(plot.ax.lines, linewidth=2)
	plot.ax.set_title("Moving Average (Humidity Data)")
	plot.ax.xaxis.set_major_locator(plt.NullLocator())
	plot.ax.xaxis.set_major_formatter(plt.NullFormatter())
	plt.legend(loc="lower center", labels=["Original", "Smoothed"])
	plot.fig.autofmt_xdate()
	plot.tight_layout()
	plot.savefig("moving_average.png")


def graph_simple_imputer():
	"""
		Graph the results of the simple imputer
	"""
	timeseries: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")
	timeseries = timeseries[["date", "Zch_Stampfenbachstrasse.Humidity"]][-750:-250]
	data = timeseries.set_index("date")
	data_imputed = impute_simple_imputer(data, False)
	
	data.columns = ["Humidity (Original)"]
	data_imputed.columns = ["Humidity (Imputed)"]
	
	data_plt = data_imputed.join(data)

	sns.set(rc={"figure.figsize": (2, 1)})
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(data=data_plt, kind="line", dashes=["", ""], legend=False, aspect=2)
	plt.setp(plot.ax.lines, linewidth=2)
	plot.ax.set_title("Simple Imputer (Humidity Data)")
	plot.ax.xaxis.set_major_locator(plt.NullLocator())
	plot.ax.xaxis.set_major_formatter(plt.NullFormatter())
	plt.legend(loc="lower center", labels=["Imputed", "Original"])
	plot.tight_layout()
	plot.savefig("simple_imputer.png")
