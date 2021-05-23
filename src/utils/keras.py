from tqdm import tqdm
from keras.callbacks import Callback


class KerasTrainCallback(Callback):
    def __init__(self, progress: tqdm):
        super().__init__()
        self.progress = progress

    def on_epoch_end(self, epoch, logs=None):
        acc, loss = logs.get('acc'), logs.get('loss')
        line = f'Epoch {epoch} - train loss: {loss:.4f}'
        if acc is not None:
            line += f' - train acc: {acc:.2%}'
        tqdm.write(line)
        self.progress.update()