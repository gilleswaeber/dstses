import sqlalchemy
from pathlib import Path

db_file = Path(__file__).parent.parent.absolute() / "hist_data/hist_data.sqlite"
headers_nabel = [ "PM10", "PM2.5", "Temperature", "Rainfall" ]
headers_zurich = [ "PM10", "PM2.5", "Humidity", "Temperature", "Pressure" ]


def get_engine() -> sqlalchemy.engine.Connection:
	"""
		Creates and returns a database engine able to connect to the 'hist_data.sqlite' file.
	"""
	return sqlalchemy.create_engine(f'sqlite:///{db_file}', echo=False)


def get_time_series(engine: sqlalchemy.engine.Connection, dataset: str, location: str):
	"""
		extracts a timeseries from the database containing exactly one series of measurements from exactly one location
	"""
	# make sure that the dataset exists
	assert dataset == 'nabel' or dataset == 'zurich'
	# prefix all headers with the specified location
	headers = list(map(lambda x: f'\'{location}.{x}\'', headers_nabel if dataset == 'nabel' else headers_zurich))
	query: str = f'SELECT {",".join(headers)} FROM {dataset} LIMIT 10;'
	print(query)
	for a in engine.execute(query).fetchall(): print(a[headers[0]])
	

# get_time_series(get_engine(), "nabel", "Dübendorf-Empa")
print(get_engine().execute("SELECT 'Dübendorf-Empa.PM10' FROM nabel LIMIT 1").fetchall())