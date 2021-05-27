from configparser import SectionProxy

import numpy as np

from utils.config import default_config


class Normalizer:
	def __init__(self, config: SectionProxy=None):
		if config is None:
			config = default_config()['normalization']
		self.pm_max = float(config['pm_max'])
		self.humidity_max = float(config['humidity_max'])
		self.pressure_min = float(config['pressure_min'])
		self.pressure_max = float(config['pressure_max'])
		self.temperature_min = float(config['temperature_min'])
		self.temperature_max = float(config['temperature_max'])

	def normalize_pm(self, x):
		return np.clip(x, 0, self.pm_max) / self.pm_max

	def normalize_pressure(self, x):
		return (np.clip(x, self.pressure_min, self.pressure_max) - self.pressure_min) / (self.pressure_max - self.pressure_min)

	def normalize_humidity(self, x):
		return np.clip(x, 0, self.humidity_max) / self.humidity_max

	def normalize_temperature(self, x):
		return (np.clip(x, self.temperature_min, self.temperature_max) - self.temperature_min) / (self.temperature_max - self.temperature_min)

	def normalize(self, x, feature):
		if feature in ('PM10', 'PM2.5'):
			return self.normalize_pm(x)
		elif feature == 'Temperature':
			return self.normalize_temperature(x)
		elif feature == 'Pressure':
			return self.normalize_pressure(x)
		elif feature == 'Humidity':
			return self.normalize_humidity(x)
		else:
			raise Exception(f'Unknown feature {feature}')