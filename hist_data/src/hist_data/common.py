from datetime import datetime
from itertools import chain

import pandas
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

from utils.config import INFLUXDB_HIST_DATA_BUCKET, HIST_DATA_DB, default_config


def get_engine() -> Connection:
	return create_engine(f'sqlite:///{HIST_DATA_DB}', echo=False)


def upload_influx_db(con: Connection, table: str, start_time: datetime = None):
	from influxdb_client import InfluxDBClient
	from influxdb_client.client.write_api import SYNCHRONOUS
	client = InfluxDBClient.from_config_file(default_config()['influx']['config'])
	write_api = client.write_api(write_options=SYNCHRONOUS)

	cols = set(con.execute(f'SELECT * FROM {table} LIMIT 1').keys())
	places = set(p.split('.', 1)[0] for p in cols)
	places_pm10 = set(p for p in places if f'{p}.PM10' in cols)
	places_pm2_5 = set(p for p in places if f'{p}.PM2.5' in cols)
	places_temperature = set(p for p in places if f'{p}.Temperature' in cols)
	places_rainfall = set(p for p in places if f'{p}.Rainfall' in cols)
	places_humidity = set(p for p in places if f'{p}.Humidity' in cols)
	places_pressure = set(p for p in places if f'{p}.Pressure' in cols)

	places = sorted(set(chain(places_pm10, places_pm2_5, places_temperature, places_rainfall)))

	q = " || '\n' || ".join(
		f"""'pollution,place={p} ' || SUBSTR(""" +
		(f"""IFNULL(',pm10=' || "{p}.PM10", '') || """ if p in places_pm10 else '') +
		(f"""IFNULL(',pm2.5=' || "{p}.PM2.5", '') || """ if p in places_pm2_5 else '') +
		f"""'', 2) || STRFTIME(' %s000000000', date) || '\n' || """ +

		f"""'weather,place={p} ' || SUBSTR(""" +
		(f"""IFNULL(',temperature=' || "{p}.Temperature", '') || """ if p in places_temperature else '') +
		(f"""IFNULL(',rainfall=' || "{p}.Rainfall", '') || """ if p in places_rainfall else '') +
		(f"""IFNULL(',pressure=' || "{p}.Pressure", '') || """ if p in places_pressure else '') +
		(f"""IFNULL(',humidity=' || "{p}.Humidity", '') || """ if p in places_humidity else '') +
		f"""'', 2) || STRFTIME(' %s000000000', date)"""

		for p in places
	)

	query = f"""SELECT {q} AS line_data FROM {table}"""
	if start_time is not None:
		query += f" WHERE date >= '{start_time.isoformat()}'"

	res = con.execute(query)

	sequence = [l for row in res for l in row['line_data'].split('\n') if '  ' not in l]

	print('SQL query:', query)
	# print('\n'.join(sequence))

	write_api.write(INFLUXDB_HIST_DATA_BUCKET, records=sequence)

	print('Done!')


def table_exists(table: str, con: Connection):
	r = con.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
	return r.first() is not None


def add_columns(table: str, df: pandas.DataFrame, con: Connection):
	"""Add columns eventually missing in the table, if the table exists"""
	if table_exists(table, con):
		cols = set(con.execute(f'SELECT * FROM {table} LIMIT 1').keys())
		new_cols = set(df.columns) - cols
		for col in new_cols:
			alter = f'ALTER TABLE {table} ADD COLUMN "{col}" FLOAT'
			con.execute(alter)
