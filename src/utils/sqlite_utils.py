from pathlib import Path

import pandas as pd
import sqlalchemy

db_file = Path(__file__).parent.parent.parent.absolute() / "hist_data/data/hist_data.sqlite"
headers_nabel = ["PM10", "PM2.5", "Temperature", "Rainfall"]
headers_zurich = ["PM10", "PM2.5", "Humidity", "Temperature", "Pressure"]


def get_engine() -> sqlalchemy.engine.Connection:
	"""
		Creates and returns a database engine able to connect to the 'hist_data.sqlite' file.
	"""
	return sqlalchemy.create_engine(f'sqlite:///{db_file}', echo=False)


def get_time_series(engine: sqlalchemy.engine.Connection, dataset: str, location: str) -> pd.DataFrame:
	"""
		extracts a timeseries from the database containing exactly one series of measurements from exactly one location
	"""
	# make sure that the dataset exists
	assert dataset == 'nabel' or dataset == 'zurich'

	# create list of headers to query
	prefixed = [f'"{location}.{x}"' for x in (headers_nabel if dataset == 'nabel' else headers_zurich)]
	headers = ",".join(['"date"'] + prefixed)
	query = f'SELECT {headers} FROM {dataset};'
	return pd.read_sql_query(query, engine)
