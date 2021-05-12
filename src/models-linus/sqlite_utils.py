import sqlalchemy
import pandas as pd
from pathlib import Path

db_file = Path(__file__).parent.parent.absolute() / "hist_data/hist_data.sqlite"
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
	headers = ",".join(['"date"'] + [f'"{location}.{x}"' for x in (headers_nabel if dataset == 'nabel' else headers_zurich)])
	query = f'SELECT {headers} FROM {dataset};'
	return pd.read_sql_query(query, engine)



#connection = get_engine().connect()
#zurich_data: pd.DataFrame = pd.read_sql_table('zurich', connection)
#nabel_data: pd.DataFrame = pd.read_sql_table('nabel', connection)

#print(nabel_data)
#print(nabel_data.filter(items=[ "date" ] + list(filter(lambda x: "Dübendorf" in x, nabel_data.columns)), axis=1))


#for row in get_engine().execute('SELECT "date","Dübendorf-Empa.PM10","Dübendorf-Empa.PM2.5","Dübendorf-Empa.Temperature","Dübendorf-Empa.Rainfall" FROM nabel LIMIT 10;').fetchall(): print(row)
#for row in get_engine().execute('SELECT "date","Zch_Stampfenbachstrasse.PM10","Zch_Stampfenbachstrasse.PM2.5","Zch_Stampfenbachstrasse.Humidity","Zch_Stampfenbachstrasse.Temperature","Zch_Stampfenbachstrasse.Pressure" FROM zurich LIMIT 10;').fetchall(): print(row)
get_time_series(get_engine(), "nabel", "Dübendorf-Empa")

