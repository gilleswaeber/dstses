import time


class Timer:
	"""
		Creates a simple timer that can be used to track the running time of a block of code.
	"""

	def __init__(self):
		"""
			Creates a new simple timer and starts it.
		"""
		self.time_start = time.time()
		self.time_stop = None

	def now(self) -> float:
		"""
			Returns the number of seconds this timer has been running up until now, or up until it was stopped.
		"""
		if self.time_stop is None:
			return time.time() - self.time_start
		else:
			return self.time_stop - self.time_start

	def stop(self) -> float:
		"""
			Stops this timer. Any future read from this timer will behave as if it happed at the time of this call.
		"""
		if self.time_stop is None:
			self.time_stop = time.time()
		return self.time_stop

	def __str__(self) -> str:
		"""
			Converts this timer into a string using the 'format_time' function. If this timer has been stopped,
			the time it was stopped will be taken into account, if it hasn't been stopped yet, the time of the
			call to this function will be taken into account.
		"""
		if self.time_stop is None:
			return format_time(time.time() - self.time_start)
		else:
			return format_time(self.time_stop - self.time_start)


def format_time(duration: float) -> str:
	"""
		Formats a duration into a string of format 'hh:mm:ss.ms'.
	"""
	hours, rem = int(duration / 3600), duration % 3600
	mins, secs = int(rem / 60), rem % 60
	hours_str = f'{hours}' if hours > 9 else f'0{hours}'
	mins_str = f'{mins}' if mins > 9 else f'0{mins}'
	secs_str = f'{secs:.2f}' if secs >= 10 else f'0{secs:.2f}'
	return f'{hours_str}:{mins_str}:{secs_str}s'
