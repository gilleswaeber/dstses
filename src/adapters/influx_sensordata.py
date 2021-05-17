from configparser import ConfigParser

from influxdb_client import InfluxDBClient

from adapters import data_adapter
from utils import logger
from pathlib import Path

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

	def get_data(self):
		with InfluxDBClient.from_config_file(
				str(Path(self.config["resources_path"]) / self.config["influx"]["config"])
		) as client:
			query = client.query_api()

			return query.query_data_frame(f'from(bucket:"{self.bucket}")'
										  f'|> range(start: {self.start})'
										  '|> filter(fn: (r) => r["_measurement"] == "pollution" or r["_measurement"] == "weather")'
										  '|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)'
										  '|> yield(name: "mean")')
