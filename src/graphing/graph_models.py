"""
	This file contains functions to graph the results of different models
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sktime.forecasting.model_selection import temporal_train_test_split
from pathlib import Path

from models import autoarima
from models import expsmoothing
from models import arima
from models import lstm

from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average

from utils.sqlite_utils import get_engine, get_time_series
from utils.config import default_config

# TODO: determine some sane forecasting horizon here
fh = 48
config = default_config()
from_excel = False


def write_model_graph(y_train, y_test, y_pred, name):
	data = np.array([
		np.append(y_train, [np.nan] * y_test.shape[0]),
		np.append([np.nan] * y_train.shape[0], y_test),
		np.append([np.nan] * y_train.shape[0], y_pred)
	]).transpose()
	
	idx = y_train.shape[0] - 1
	data[idx][1] = data[idx][0]
	data[idx][2] = data[idx][0]
	
	palette = sns.color_palette(["#0000ff", "#00bb00", "#ff0000"])
	sns.set_theme(style="darkgrid")
	plot = sns.relplot(data=pd.DataFrame(data), kind="line", dashes=["", "", ""], palette=palette, legend=False, aspect=1.5)
	plot.set(xlabel="Time", ylabel="PM10 [µg/m³]", title=f"{name} {fh}h forecast")
	plot.ax.xaxis.set_major_locator(plt.NullLocator())
	plot.ax.xaxis.set_major_formatter(plt.NullFormatter())
	plt.legend(loc="lower center", labels=["Training Phase", "Ground Truth", "Prediction"])
	plot.tight_layout()
	suffix = "_excel" if from_excel else ""
	plot.savefig(f"{name.lower().replace(' ', '_')}{suffix}.png")


def graph_model_autoarima():
	if from_excel:
		y_train, y_test, y_pred = get_data_from_excel("AutoArima.PM10")
		write_model_graph(y_train, y_test, y_pred, "AutoARIMA")
	else:
		ts: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")[-1100:-900]
		ts.drop(columns=["date", "Zch_Stampfenbachstrasse.PM2.5"], inplace=True)
		ts_imputed = impute_simple_imputer(ts, False)
		ts_smooth = moving_average(ts_imputed, False)
		y, x = autoarima.transform_data(ts_smooth, False)
		
		y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=fh)
		model = autoarima.train_model_autoarima(y_train, x_train, False)
		y_pred = model.predict(X=x_test, fh=np.linspace(1, fh, fh))
		
		write_model_graph(y_train, y_test, y_pred, "AutoARIMA")


def graph_model_arima():
	if from_excel:
		y_train, y_test, y_pred = get_data_from_excel("Arima.PM10")
		write_model_graph(y_train, y_test, y_pred, "ARIMA")
	else:
		ts: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")[-1100:-900]
		ts.drop(columns=["date", "Zch_Stampfenbachstrasse.PM2.5"], inplace=True)
		ts_imputed = impute_simple_imputer(ts, False)
		ts_smooth = moving_average(ts_imputed, False)
		y, x = arima.transform_data(ts_smooth, False)
	
		y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=fh)
		model = arima.train_model_arima(y_train, x_train, False)
		y_pred = model.predict(X=x_test, fh=np.linspace(1, fh, fh))
	
		write_model_graph(y_train, y_test, y_pred, "ARIMA")


def graph_model_exp_smoothing():
	if from_excel:
		y_train, y_test, y_pred = get_data_from_excel("ExpSmoothing.PM10")
		write_model_graph(y_train, y_test, y_pred, "Exponential Smoothing")
	else:
		ts: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")[-1100:-900]
		ts.drop(columns=["date", "Zch_Stampfenbachstrasse.PM2.5"], inplace=True)
		ts_imputed = impute_simple_imputer(ts, False)
		ts_smooth = moving_average(ts_imputed, False)
		y, x = expsmoothing.transform_data(ts_smooth, False)
		
		y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=fh)
		model = expsmoothing.train_model_expSmooting(y_train, x_train, False)
		y_pred = model.predict(X=x_test, fh=np.linspace(1, fh, fh))
		
		write_model_graph(y_train, y_test, y_pred, "Exponential Smoothing")


def graph_model_lstm():
	if from_excel:
		y_train, y_test, y_pred = get_data_from_excel("LSTM.PM10")
		write_model_graph(y_train, y_test, y_pred, "LSTM")
	else:
		y_train, y_test, y_pred = get_data_from_excel("LSTM.PM10")
		write_model_graph(y_train, y_test, y_pred, "LSTM")


def graph_model_lstmseq():
	if from_excel:
		y_train, y_test, y_pred = get_data_from_excel("LSTMSeq.PM10")
		write_model_graph(y_train, y_test, y_pred, "LSTMSeq")
	else:
		y_train, y_test, y_pred = get_data_from_excel("LSTMSeq.PM10")
		write_model_graph(y_train, y_test, y_pred, "LSTMSeq")
	
	
def get_data_from_excel(pred_name):
	path = Path(__file__).parent.parent.parent / "prediction_data/PM10.xlsx"
	ts: pd.DataFrame = pd.read_excel(io=path, sheet_name="graph")
	
	pred_start = ts["LSTM.PM10"].first_valid_index()
	start = pred_start - (200 - fh) if not from_excel else 0
	stop = pred_start + fh if not from_excel else ts.shape[0]
	
	y_train = ts["Live.PM10"][start:pred_start]
	y_test = ts["Live.PM10"][pred_start:stop]
	y_pred = ts[pred_name][pred_start:stop]
	
	return y_train, y_test, y_pred
