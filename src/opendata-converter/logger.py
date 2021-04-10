from sys import stdout, stderr
from datetime import datetime
from io import TextIOWrapper



"""
	Simple logger class to provide simple logging capabilities.
	This class supports a total of six logging levels (FATAL, ERROR, WARNING, INFO, DEBUG, VERBOSE) and an arbitrary amount of output streams.
	Each output stream can be annotated with a min and max logging level that it should output. This is useful for filtering error logs and such.
"""
class Logger:
	
	LEVEL_FATAL: int = 0
	LEVEL_ERROR: int = 1
	LEVEL_WARNING: int = 2
	LEVEL_INFO: int = 3
	LEVEL_DEBUG: int = 4
	LEVEL_VERBOSE: int = 5
	
	# lists of outputstreams for where each message should be printed
	__out_streams = [ (stdout, 2, 5), (stderr, 0, 1) ]
	
	"""
		Creates a new instance of this Logger class. Every instance has a 'module name' associated with it,
		that will be printed before each message
	"""
	def __init__(self, module_name: str):
		self.module_name = module_name
	
	@staticmethod
	def __prefix() -> str:
		return datetime.now().strftime("[%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
	
	@staticmethod
	def __level_name(level: int) -> str:
		if level == Logger.LEVEL_FATAL: return "[FATAL]"
		if level == Logger.LEVEL_ERROR: return "[ERROR]"
		if level == Logger.LEVEL_WARNING: return "[WARNING]"
		if level == Logger.LEVEL_INFO: return "[INFO]"
		if level == Logger.LEVEL_DEBUG: return "[DEBUG]"
		if level == Logger.LEVEL_VERBOSE: return "[VERBOSE]"
		return "[UNKNOWN]"
	
	"""
		Adds a new normal output to the logging class. A normal output will log all messages no matter what logging level they are.
	"""
	@staticmethod
	def addNormalOutput(stream: TextIOWrapper) -> None:
		Logger.__out_streams.append( (stream, 0, 5) )
	
	"""
		Adds a new error output to the logging class. An error output will log any messages with a logging level of ERROR or FATAL.
	"""
	@staticmethod
	def addErrorOutput(stream: TextIOWrapper) -> None:
		Logger.__out_streams.append( (stream, 0, 1) )
	
	"""
		Adds a new output to the logging class. The min_level and max_level arguments specify the lowest and the highest level that this output will accept.
	"""
	@staticmethod
	def addCustomOutput(stream: TextIOWrapper, min_level: int, max_level: int) -> None:
		Logger.__out_streams.append( (stream, min_level, max_level) )
	
	def __log(self, level: int, msg: str) -> None:
		for (stream, min_level, max_level) in Logger.__out_streams:
			if min_level <= level and max_level >= level:
				print(Logger.__prefix() + Logger.__level_name(level) + " " + msg, file=stream)
	
	"""
		Logs a message of level FATAL.
	"""
	def fatal(self, msg: str) -> None:
		self.__log(Logger.LEVEL_FATAL, msg)
	
	"""
		Logs a message of level ERROR.
	"""
	def error(self, msg: str) -> None:
		self.__log(Logger.LEVEL_ERROR, msg)
	
	"""
		Logs a message of level WARNING.
	"""
	def warn(self, msg: str) -> None:
		self.__log(Logger.LEVEL_WARNING, msg)
	
	"""
		Logs a message of level INFO.
	"""
	def info(self, msg: str) -> None:
		self.__log(Logger.LEVEL_INFO, msg)
	
	"""
		Logs a message of level DEBUG.
	"""
	def debug(self, msg: str) -> None:
		self.__log(Logger.LEVEL_DEBUG, msg)
	
	"""
		Logs a message of level VERBOSE.
	"""
	def verbose(self, msg: str) -> None:
		self.__log(Logger.LEVEL_VERBOSE, msg)
	
	