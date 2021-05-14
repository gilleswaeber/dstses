from sys import stdout, stderr
from datetime import datetime
from io import TextIOWrapper


class Logger:
	"""
		Simple logger class to provide simple logging capabilities.
		This class supports a total of six logging levels (FATAL, ERROR, WARNING, INFO, DEBUG, VERBOSE) and an arbitrary
		amount of output streams. Each output stream can be annotated with a min and max logging level that it should
		output. This is useful for filtering error logs and such.
	"""

	LEVEL_FATAL: int = 0
	LEVEL_ERROR: int = 1
	LEVEL_WARNING: int = 2
	LEVEL_INFO: int = 3
	LEVEL_DEBUG: int = 4
	LEVEL_VERBOSE: int = 5

	# lists of output streams for where each message should be printed
	__out_streams = [(stdout, 2, 5), (stderr, 0, 1)]
	__progress = None
	__console_width = 120

	def __init__(self, module_name: str):
		"""
			Creates a new instance of this Logger class. Every instance has a 'module name' associated with it,
			that will be printed before each message
		"""
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

	@staticmethod
	def addNormalOutput(stream: TextIOWrapper) -> None:
		"""
			Adds a new normal output to the logging class. A normal output will log all messages no matter what logging
			level they are.
		"""
		Logger.__out_streams.append((stream, 0, 5))

	@staticmethod
	def addErrorOutput(stream: TextIOWrapper) -> None:
		"""
			Adds a new error output to the logging class. An error output will log any messages with a logging level of
			ERROR or FATAL.
		"""
		Logger.__out_streams.append((stream, 0, 1))

	@staticmethod
	def addCustomOutput(stream: TextIOWrapper, min_level: int, max_level: int) -> None:
		"""
			Adds a new output to the logging class. The min_level and max_level arguments specify the lowest and the
			highest level that this output will accept.
		"""
		Logger.__out_streams.append((stream, min_level, max_level))

	def __log(self, level: int, msg: str, state: str = None) -> None:
		for (stream, min_level, max_level) in Logger.__out_streams:
			if min_level <= level <= max_level:
				line_pre = f"{Logger.__prefix()}{Logger.__level_name(level)}[{self.module_name}] "
				remaining_length = Logger.__console_width - len(line_pre)
				if state == "begin" and stream is stdout:
					Logger.__progress = msg
					print(f'{line_pre}{Logger.__progress}', file=stream, end="", flush=True)
				elif state == "update" and stream is stdout:
					width = remaining_length - len(Logger.__progress) - len(msg)
					out = f'{"".join([" "] * width)}{msg}'
					print(f'\r{line_pre}{Logger.__progress}{out}', file=stream, end="", flush=True)
				elif state == "end" and stream is stdout:
					width = remaining_length - len(Logger.__progress) - len(msg)
					out = f'{"".join([" "] * width)}{msg}'
					print(f"\r{line_pre}{Logger.__progress}{out}", file=stream, flush=True)
					Logger.__progress = None
				elif state is None:
					print(Logger.__prefix() + Logger.__level_name(level) + f"[{self.module_name}] {msg}", file=stream)

	def fatal(self, msg: str) -> None:
		"""
			Logs a message of level FATAL.
		"""
		self.__log(Logger.LEVEL_FATAL, msg)

	def error(self, msg: str) -> None:
		"""
			Logs a message of level ERROR.
		"""
		self.__log(Logger.LEVEL_ERROR, msg)

	def warn(self, msg: str) -> None:
		"""
			Logs a message of level WARNING.
		"""
		self.__log(Logger.LEVEL_WARNING, msg)

	def info(self, msg: str) -> None:
		"""
			Logs a message of level INFO.
		"""
		self.__log(Logger.LEVEL_INFO, msg)

	def debug(self, msg: str) -> None:
		"""
			Logs a message of level DEBUG.
		"""
		self.__log(Logger.LEVEL_DEBUG, msg)

	def verbose(self, msg: str) -> None:
		"""
			Logs a message of level VERBOSE.
		"""
		self.__log(Logger.LEVEL_VERBOSE, msg)

	def info_begin(self, msg: str) -> None:
		"""
			Begins a new progress bar on stdout. This only works on stdout.
			Also this does not work in a context where multiple entities concurrently
			write to stdout.
		"""
		assert Logger.__progress is None
		self.__log(Logger.LEVEL_INFO, msg, state="begin")

	def info_update(self, msg: str) -> None:
		"""
			Updates a previously created progress on stdout
		"""
		assert Logger.__progress is not None
		self.__log(Logger.LEVEL_INFO, msg, state="update")

	def info_end(self, msg: str) -> None:
		"""
			Ends a previously created progress on stdout
		"""
		assert Logger.__progress is not None
		self.__log(Logger.LEVEL_INFO, msg, state="end")
