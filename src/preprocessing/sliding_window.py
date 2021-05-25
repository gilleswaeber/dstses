import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
	"""Taken from numpy v1.20,
	See: https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html"""
	window_shape = (tuple(window_shape)
					if np.iterable(window_shape)
					else (window_shape,))
	# first convert input to array, possibly keeping subclass
	x = np.array(x, copy=False, subok=subok)

	window_shape_array = np.array(window_shape)
	if np.any(window_shape_array < 0):
		raise ValueError('`window_shape` cannot contain negative values')

	if axis is None:
		axis = tuple(range(x.ndim))
		if len(window_shape) != len(axis):
			raise ValueError(f'Since axis is `None`, must provide '
							 f'window_shape for all dimensions of `x`; '
							 f'got {len(window_shape)} window_shape elements '
							 f'and `x.ndim` is {x.ndim}.')
	else:
		axis = np.core.numeric.normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
		if len(window_shape) != len(axis):
			raise ValueError(f'Must provide matching length window_shape and '
							 f'axis; got {len(window_shape)} window_shape '
							 f'elements and {len(axis)} axes elements.')

	out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

	# note: same axis can be windowed repeatedly
	x_shape_trimmed = list(x.shape)
	for ax, dim in zip(axis, window_shape):
		if x_shape_trimmed[ax] < dim:
			raise ValueError(
				'window shape cannot be larger than input array shape')
		x_shape_trimmed[ax] -= dim - 1
	out_shape = tuple(x_shape_trimmed) + window_shape
	return as_strided(x, strides=out_strides, shape=out_shape,
					  subok=subok, writeable=writeable)


def slide_rows(data: np.ndarray, length: int):
	return np.swapaxes(sliding_window_view(data, window_shape=length, axis=0), 1, 2)


def prepare_window_next(data: np.ndarray, length: int):
	"""Prepare a dataset for training using a sliding window

	Input: [n, f] array
	Output: x=[n - v + 1, l, f], y=[n - v + 1, 1, f] arrays

	Where:
	- n is the number of samples
	- v is the number of feature vectors in each sample
	- f is the number of features
	- l is the window_length
	"""
	sliding_window = np.swapaxes(sliding_window_view(data, window_shape=length, axis=0), 1, 2)
	y = sliding_window[:, -1:, :]  # use last value as y
	x = sliding_window[:, :-1, :]  # take the rest as x

	return x, y


def prepare_window_off_by_1(data: np.ndarray, length: int):
	"""Prepare a dataset for training using a sliding window

	Input: [n, f] array
	Output: x=[n - v + 1, l, f], y=[n - v + 1, l, f] arrays

	Where:
	- n is the number of samples
	- v is the number of feature vectors in each sample
	- f is the number of features
	- l is the window_length
	- y[i,j,k] = x[i,j-1,k]
	"""
	x = np.swapaxes(sliding_window_view(data[:-1], window_shape=length, axis=0), 1, 2)
	y = np.swapaxes(sliding_window_view(data[1:], window_shape=length, axis=0), 1, 2)

	return x, y
