from configparser import ConfigParser
from datetime import datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from adapters import data_adapter
from utils import logger
from utils.config import default_config

"""
	Loads the influx db server data and fills the data into the dataframe.
	This class needs the configurtion of Influxdbclient in the config file, for more see:
	https://github.com/influxdata/influxdb-client-python
	
	and the following config options:
	config[influx][config] - Path to the config file
	config[influx][name][bucket] - the bucket to download
	config[influx][name][start] - (optional = -30d) 
	
	The client calls influxdb and downloads the requested timeframe and aggregates all data to 1h / datapoint to reduce
	data-transfer and speedup computation afterwards.
"""


class InfluxSensorData(data_adapter.IDataAdapter):

	def __init__(self, config: ConfigParser, name):
		super().__init__(logger.Logger(module_name=f"influx data adapter '{name}'"), config)
		self.name = name
		self.bucket = config[name]["bucket"]
		self.start = config[name].get("start", "-30d")
		self.client = InfluxDBClient.from_config_file(self.config["influx"]["config"], debug=True)

	def get_data(self):
		query = self.client.query_api()

		return query.query_data_frame(f'from(bucket:"{self.bucket}")'
									  f'|> range(start: {self.start})'
									  '|> filter(fn: (r) => r["_measurement"] == "pollution" or r["_measurement"] == "weather")'
									  '|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)'
									  '|> sort(columns: ["_time"], desc: false)'
									  '|> yield(name: "mean")')

	def send_data(self, value):
		point = Point("prediction").field("prediction", value).time(datetime.now())
		with self.client.write_api(write_options=SYNCHRONOUS) as write_api:
			write_api.write(point)

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.client.close()


if __name__ == '__main__':
	conf = default_config()
	conf['DEFAULT']['bucket'] = 'hist_data'
	conf['DEFAULT']['start'] = '-60d'
	adapt = InfluxSensorData(conf, 'DEFAULT')
	data = adapt.get_data()
	print(data)