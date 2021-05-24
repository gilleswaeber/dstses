"""
	This file contains functions to graph the results of different models
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split

from models import autoarima
from preprocessing.imputing import impute_simple_imputer
from preprocessing.moving_average import moving_average
from utils.sqlite_utils import get_engine, get_time_series


def graph_model_autoarima():
	ts: pd.DataFrame = get_time_series(get_engine(), "zurich", "Zch_Stampfenbachstrasse")[-1100:-900]
	ts.drop(columns=["date"], inplace=True)
	ts_imputed = impute_simple_imputer(ts, False)
	ts_smooth = moving_average(ts_imputed, False)
	y, x = autoarima.transform_data(ts_smooth, False)

	fh = 24  # TODO: determine sane fh
	y_train, y_test, x_train, x_test = temporal_train_test_split(y, x, test_size=fh)
	model = autoarima.train_model_autoarima(y_train, x_train, False)
	y_pred = model.predict(X=x_test, fh=np.linspace(1, fh, fh))

	fig, ax = plt.subplots(1)
	ax.set_title(f"AutoARIMA model, {fh}h forecast")
	ax.plot(y_train, color='blue')
	ax.plot(y_test, color='green')
	ax.plot(y_pred, color='red')
	fig.savefig("autoarima.png")

	print(y_pred.shape, x_test.shape, y_test.shape)
