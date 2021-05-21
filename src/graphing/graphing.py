"""
	This file is responsible for drawing graphs of our data.
"""

import sys
import os

from typing import Callable, Dict

from graph_data import graph_random_noise
from graph_preprocessing import graph_moving_average, graph_simple_imputer
from graph_models import graph_model_autoarima

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
		if arg == 'all':
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
		'random': Command('random', 'Graphs random noise', graph_random_noise),
		'moving_average': Command('moving_average', 'Graphs the effects of moving average', graph_moving_average),
		'simple_imputer': Command('simple_imputer', 'Graphs the effects of a simple imputer', graph_simple_imputer),
		'autoarima': Command('autoarima', 'Graphs the predictions of AutoARIMA', graph_model_autoarima),
	}
	main(cmds)
