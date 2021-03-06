"""
	This module collects functions used to score a model.
"""
import numpy as np
from sktime.performance_metrics.forecasting import mape_loss

from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Evaluation")


def eval_model_mape(model, y_test, x_test, output: bool = True) -> float:
	if output:
		logger.info_begin("Measuring performance metrics...")
		timer = Timer()
		logger.info_update("Computing")
	fh = np.arange(1, x_test.shape[0] + 1)
	y_pred = model.predict(X=x_test, fh=fh)
	if output:
		logger.info_update("Scoring")
	y_pred.index = range(len(y_pred))
	logger.info(y_pred)
	error = mape_loss(y_test, y_pred)
	if output:
		logger.info_end(f'Done in {timer}')
	return error
