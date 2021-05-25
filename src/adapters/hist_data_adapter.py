from configparser import ConfigParser

import pandas as pd
import sqlalchemy
from sqlalchemy.engine import Connection

from adapters import data_adapter
from utils import logger
from utils.config import HIST_DATA_DB

"""
	Opens the local sqlite database downloaded by hist_data and fills the database into dataframe. Needs as optional
	parameter the table name in the constructor.
	If no dataset and/or location was given, it will load them from the application configuration to make changing
	even simpler. If given it will use the given configuration.
"""

HEADERS_NABEL = ["PM10", "PM2.5", "Temperature", "Rainfall"]
HEADERS_ZURICH = ["PM10", "PM2.5", "Humidity", "Temperature", "Pressure"]


class HistDataAdapter(data_adapter.IDataAdapter):
	def __init__(self, config: ConfigParser, name, dataset=None, location=None):
		super().__init__(logger.Logger(module_name=f"hist data adapter '{name}'"), config)
		self.name = name
		self.dataset = config[name]["dataset"] if dataset is None else dataset
		self.location = config[name]["location"] if location is None else location

	def get_engine(self) -> sqlalchemy.engine.Connection:
		"""
			Creates and returns a database engine able to connect to the 'hist_data.sqlite' file.
		"""
		return sqlalchemy.create_engine(f'sqlite:///{HIST_DATA_DB}', echo=False)

	def get_time_series(self, engine: sqlalchemy.engine.Connection, dataset: str, location: str) -> pd.DataFrame:
		"""
			extracts a timeseries from the database containing exactly one series of measurements from exactly one location
		"""
		# make sure that the dataset exists
		assert dataset in ('nabel', 'zurich')

		# create list of headers to query
		prefixed = self.loc_columns(dataset, location)
		headers = ",".join(['"date"'] + prefixed)
		query = f'SELECT {headers} FROM {dataset} ORDER BY date'
		return pd.read_sql_query(query, engine)

	def get_multiple_locations(self, dataset, locations):
		# make sure that the dataset exists
		assert dataset in ('nabel', 'zurich')


	@staticmethod
	def loc_columns(dataset, location):
		assert dataset in ('nabel', 'zurich')
		return [f'"{location}.{x}"' for x in (HEADERS_NABEL if dataset == 'nabel' else HEADERS_ZURICH)]

	@staticmethod
	def table_exists(table, con: Connection):
		r = con.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
		return r.first() is not None

	def get_data(self):
		self.logger.info(f"Loading sqlite...")
		with self.get_engine().connect() as con:
			assert self.table_exists(self.dataset, con)
			table = self.get_time_series(con, self.dataset, self.location)
			self.logger.info(f"Loading sqlite... done")
			return table
