"""
	This file contains some functions that graph datasets, either loaded or generated.
"""

import matplotlib.pyplot as plt
import numpy as np


def graph_random_noise():
	"""
		Graphs 100 steps of a randomly generated timeseries
	"""
	data = np.random.random(100)

	fig, ax = plt.subplots(1)
	ax.set_title("Random Noise")
	ax.plot(data)
	fig.savefig("random_noise.png")
