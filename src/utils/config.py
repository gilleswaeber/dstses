import configparser
from configparser import ConfigParser
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent

RESOURCE_DIR = PROJECT_DIR / 'src/resources'

HIST_DATA_DB = PROJECT_DIR / 'hist_data/data' / 'hist_data.sqlite'

CONFIG_FILE = RESOURCE_DIR / 'config.ini'
CONFIG_INFLUXDB = RESOURCE_DIR / 'influx2.ini'

INFLUXDB_HIST_DATA_BUCKET = 'hist_data'


def default_config() -> ConfigParser:
	print(CONFIG_FILE)
	config = ConfigParser()
	config._interpolation = configparser.ExtendedInterpolation()
	config['DEFAULT']['resources_path'] = str(RESOURCE_DIR.absolute())
	config.read(str(CONFIG_FILE))
	return config
