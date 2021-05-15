
from utils import logger
import pandas as pd
from adapters import data_adapter
from pathlib import Path
import pandas
from datetime import datetime
from itertools import chain
import sqlalchemy
from sqlalchemy.engine import Connection
from configparser import ConfigParser

"""
	Opens the local sqlite database downloaded by hist_data and fills the database into dataframe. Needs as optional
	parameter the table name in the constructor.
	If no dataset and/or location was given, it will load them from the application configuration to make changing
	even simpler. If given it will use the given configuration.
"""

class HistDataAdapter(data_adapter.IDataAdapter):
	HIST_DATA_SQLITE_DB = Path(__file__).parent.parent.parent.absolute() / "hist_data" / "data" / 'hist_data.sqlite'
	headers_nabel = ["PM10", "PM2.5", "Temperature", "Rainfall"]
	headers_zurich = ["PM10", "PM2.5", "Humidity", "Temperature", "Pressure"]

	def __init__(self, log: logger.Logger, config: ConfigParser, name, dataset = None, location = None):
		super().__init__(log, config)
		self.name = name
		self.dataset = config["sql_adapter"][name]["dataset"] if dataset is None else dataset
		self.location = config["sql_adapter"][name]["location"] if location is None else location

	def get_engine(self) -> sqlalchemy.engine.Connection:
		"""
			Creates and returns a database engine able to connect to the 'hist_data.sqlite' file.
		"""
		return sqlalchemy.create_engine(f'sqlite:///{self.HIST_DATA_SQLITE_DB}', echo=False)

	def get_time_series(self, engine: sqlalchemy.engine.Connection, dataset: str, location: str) -> pd.DataFrame:
		"""
			extracts a timeseries from the database containing exactly one series of measurements from exactly one location
		"""
		# make sure that the dataset exists
		assert dataset == 'nabel' or dataset == 'zurich'

		# create list of headers to query
		prefixed = [f'"{location}.{x}"' for x in (self.headers_nabel if dataset == 'nabel' else self.headers_zurich)]
		headers = ",".join(['"date"'] + prefixed)
		query = f'SELECT {headers} FROM {dataset};'
		return pd.read_sql_query(query, engine)

	@staticmethod
	def table_exists(table, con: Connection):
		r = con.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
		return r.first() is not None

	def get_data(self):
		self.logger.info_begin(f"dataloader {self.name} loading sqlite db")
		with self.get_engine().connect() as con:
			assert self.table_exists(self.dataset, con)
			table = self.get_time_series(con, self.dataset, self.location)
			self.logger.info_end(f"dataloader {self.name} loading sqlite db")
			return table
