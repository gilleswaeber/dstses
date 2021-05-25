from configparser import ConfigParser


class ModelHolder:
	def __init__(self, name: str, trainer: any, config: ConfigParser):
		self.name = name
		self.trainer = trainer
		self.config = config[name]