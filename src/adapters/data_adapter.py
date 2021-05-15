from utils import logger
from pandas import DataFrame
from configparser import *

"""
	IDataAdapter is the interface of any class that is capable of downloading/defining data usable in model 
	learning/testing/running purposes.
"""
class IDataAdapter:
	def __init__(self, logger: logger.Logger, config: ConfigParser):
		self.logger = logger
		self.config = config


	def get_data(self):
		"""
			Acquires the data used to learn afterwards.
			
			No parameters, returns DataFrame with all information
		"""
		self.logger.warn("Unimplemented IDataAdapter, please implement by overriding")
		return DataFrame()
