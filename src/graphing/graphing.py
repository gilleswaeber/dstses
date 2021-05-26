"""
	This file is responsible for drawing graphs of our data.
	
	Short tutorial:
		This file acts as the main module for the graphing framework. When run from the
		command line, the commands registered in the if statement at the end of the file
		are possible arguments that will perform their specified action.
		
		If this file is run in PyCharm, it will execute the command 'all', which will execute all
		other registered commands.
		
		Note: This file needs to be run either with the root directory of this project, or the 'graphing'
		package as working directory. This is because the script needs to know the relative location of
		the saved_graphs subfolder so that it can change the working directory to there.
		
		To register further commands, take a look at the Command class below. The 'name' and the 'desc'
		parameter to the constructor are helpers; any unknown command will as a result print the list of
		command names and their description to the console. The 'func' parameter must be a callable object
		that takes no arguments and does not need to return a value. This is typically a function (I recommend
		to write that function in another file, or it will get very crowded here), that performs some action,
		like graphing some data.
		
		A new command can be registered as an object of type 'Command' and entered in the map in the
		if statement, the same way there already are some commands. The callable passed as func should
		do the following few steps:
			1: produce some sort of data and graph it.
			2: save the graph into the working directory
		
		If you still have any questions, go ask Linus
"""

import os
import sys
from typing import Callable, Dict

from graph_data import graph_random_noise
from graph_data import graph_comparison_our_vs_hist
from graph_models import graph_model_autoarima
from graph_models import graph_model_arima
from graph_models import graph_model_exp_smoothing
from graph_preprocessing import graph_moving_average
from graph_preprocessing import graph_simple_imputer
from utils.logger import Logger
from utils.timer import Timer

logger = Logger("Graphing")


class Command:

	def __init__(self, name: str, desc: str, func: Callable[[], None]):
		self.name = name
		self.desc = desc
		self.func = func

	def __call__(self, *args, **kwargs):
		self.func()

	def __str__(self):
		padding = "".join([" " for _ in range(24 - len(self.name))])
		return f"{self.name}{padding}{self.desc}"


def print_header():
	logger.info("###################################################")
	logger.info("#  Datascience for Techno-Socio-Economic Systems  #")
	logger.info("#                Graphing Utility                 #")
	logger.info("###################################################")
	logger.info("")


def main(commands: Dict[str, Command]):
	print_header()

	if os.getcwd().endswith('graphing'):
		os.chdir('../..')
	elif not os.getcwd().endswith('dstses'):
		logger.error(f"ERROR: This script should only be run from the root folder of the project")
		return

	if not os.path.isdir("saved_graphs"):
		os.mkdir("saved_graphs")
	os.chdir("saved_graphs")

	# create a temporary list of all functions to be called, so as not to clutter the output if a predictable error occurs
	to_be_called = []

	for arg in sys.argv[1:]:
		if arg == 'all' or arg == 'help':
			to_be_called = list(filter(lambda x: x.name != 'all', commands.values()))
			break
		elif arg in commands:
			to_be_called.append(commands[arg])
		else:
			logger.info("usage: graphing <command>...")
			logger.info("")
			logger.info("Commands:")
			for cmd in commands.values():
				logger.info(f"\t{cmd}")
			sys.exit()

	if len(to_be_called) < 1:
		to_be_called = list(filter(lambda x: x.name != 'all', commands.values()))

	for cmd in to_be_called:
		timer = Timer()
		logger.info_begin(f"Running '{cmd.name}'...")
		cmd()
		logger.info_end(f"Done in {timer}")


if __name__ == "__main__":
	cmds = {
		'all': Command('all', 'Graph all targets', lambda: None),
		'help': Command('help', 'Prints this help page', lambda: None),
		'random': Command('random', 'Graphs random noise', graph_random_noise),
		'moving_average': Command('moving_average', 'Graphs the effects of moving average', graph_moving_average),
		'simple_imputer': Command('simple_imputer', 'Graphs the effects of a simple imputer', graph_simple_imputer),
		'autoarima': Command('autoarima', 'Graphs the predictions of AutoARIMA', graph_model_autoarima),
		'arima': Command('arima', 'Graphs the predictions of ARIMA', graph_model_arima),
		'exp_smoothing': Command('exp_smoothing', 'Graphs the predictions of exp_smoothing', graph_model_exp_smoothing),
		'our_hist': Command('our_hist', 'Graphs comparison of our data to historical data', graph_comparison_our_vs_hist)
	}
	main(cmds)
