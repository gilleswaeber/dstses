from tensorflow.keras.callbacks import Callback
from tqdm import tqdm


class KerasTrainCallback(Callback):
	def __init__(self, name: str, progress: tqdm):
		super().__init__()
		self.name = name
		self.progress = progress

	def on_epoch_end(self, epoch, logs=None):
		acc, loss = logs.get('acc'), logs.get('loss')
		val_acc, val_loss = logs.get('val_acc'), logs.get('val_loss')
		line = f'\n{self.name}: Epoch {epoch + 1} - train loss: {loss:.4g}'
		if acc is not None:
			line += f' acc: {acc:.2%}'
		if val_loss is not None:
			line += f' - validation loss: {val_loss:.4g}'
			if val_acc is not None:
				line += f' acc: {val_acc:.2%}'
		print(line)
		self.progress.update()
