import json
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
		self.limit = config[name].get("limit", "100")
		self.drops = json.loads(config[name].get("drops", '["pm1", "pm2.5", "pm4.0", "result", "table", "_time"]'))
		self.client = InfluxDBClient.from_config_file(self.config["influx"]["config"], debug=False)

	def get_data(self):
		query = self.client.query_api()

		return query.query_data_frame('import "date"// Import date library\n'
									  'import "strings" // Import string library\n'
									  f'from(bucket:"{self.bucket}")// From actual_weather_data table \n'
									  f'|> range(start: {self.start}) // select from start onwards \n'
									  '|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)'
									  '//Make sure only to take columns we can handle\n'
									  '|> filter(fn: (r) => r._measurement == "weather" or r._measurement == "pollution" and r["_field"] =~ /[a-zA-Z0-9.]*pm[a-zA-Z0-9.]*/)\n'
									  '|> drop(columns: ["place", "sensor", "_start", "_stop"]) // Throw away any columns generated above we don\'t need \n'
									  '// Cut precision in timestamp to 1m as otherwise records would not be recognized as being connected \n'
									  '|> map(fn: (r) => ({\n'
									  'r with\n'
									  '_time: date.truncate(t: r._time, unit: 1m)\n'
									  '}))\n'
									  '|> group(columns: ["_time", "_measurement", "_value"], mode: "except")// Throw away columns except the mentioned \n'
									  '|> sort(columns: ["_time"])// After group we need to sort as group leaves the field unsorted \n'
									  '|> fill(usePrevious: true) // Fill missing values when record is available but not value \n'
									  '|> map(fn: (r) => ({r with device: strings.joinStr(arr:["device", r.device], v:"")})) // Change device column to string as necessary for next step \n'
									  '|> pivot(rowKey: ["_time", "_field"], columnKey: ["device"], valueColumn: "_value") // Aggregates the rows into table with device tag as columns in new table \n'
									  '|> map(fn: (r) => ({r with _value: (r.device2 + r.device3)/2.0})) // Calculates the mean of two device columns and throws the intermediary columns away \n'
									  '|> drop(columns: ["device2", "device3", "null"])\n'
									  '|> pivot(columnKey: ["_field"], rowKey: ["_time"], valueColumn: "_value") // Instread of having sequence of fields get table of entries\n'
									  f'|> limit(n: {self.limit})// debug remove me after \n'
									  '|> sort(columns: ["_time"]) // Before exit always sort as we want to have a timeline \n'
									  '|> yield()\n'
									  ).drop(labels=self.drops, axis=1)

	def get_data_8610(self):
		query = self.client.query_api()

		return query.query_data_frame('import "date"// Import date library\n'
									  'import "strings" // Import string library\n'
									  f'from(bucket:"{self.bucket}")// From actual_weather_data table \n'
									  f'|> range(start: {self.start}) // select from start onwards \n'
									  '|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)'
									  '//Make sure only to take columns we can handle\n'
									  '|> filter(fn: (r) => r._measurement == "weather" or r._measurement == "pollution" and r["_field"] =~ /[a-zA-Z0-9.]*pm[a-zA-Z0-9.]*/)\n'
									  '|> drop(columns: ["place", "sensor", "_start", "_stop"]) // Throw away any columns generated above we don\'t need \n'
									  '// Cut precision in timestamp to 1m as otherwise records would not be recognized as being connected \n'
									  '|> map(fn: (r) => ({\n'
									  'r with\n'
									  '_time: date.truncate(t: r._time, unit: 1m)\n'
									  '}))\n'
									  '|> group(columns: ["_time", "_measurement", "_value"], mode: "except")// Throw away columns except the mentioned \n'
									  '|> sort(columns: ["_time"])// After group we need to sort as group leaves the field unsorted \n'
									  '|> fill(usePrevious: true) // Fill missing values when record is available but not value \n'
									  '|> map(fn: (r) => ({r with device: strings.joinStr(arr:["device", r.device], v:"")})) // Change device column to string as necessary for next step \n'
									  '|> pivot(columnKey: ["_field"], rowKey: ["_time"], valueColumn: "_value") // Instread of having sequence of fields get table of entries\n'
									  '|> filter(fn: (r) => r.device == "device3")'
									  f'|> limit(n: {self.limit})// debug remove me after \n'
									  '|> sort(columns: ["_time"]) // Before exit always sort as we want to have a timeline \n'
									  '|> yield()\n'
									  ).drop(labels=self.drops, axis=1)

	def send_data(self, value):
		point = Point("prediction").field("prediction", value).time(datetime.now())
		with self.client.write_api(write_options=SYNCHRONOUS) as write_api:
			write_api.write(point)

	def __enter__(self):
		self.client = InfluxDBClient.from_config_file(self.config["influx"]["config"],
													  debug=self.config[self.name]["debug"].lower() in ['true', 't',
																										'yes'])
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
