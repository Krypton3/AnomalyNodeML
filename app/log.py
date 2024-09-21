import tensorflow as tf


class LogFile(tf.keras.callbacks.Callback):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        with open(self.file_name, 'a') as f:
            f.write(f"Epoch {epoch + 1} - " +
                    f"accuracy: {logs['accuracy']:.4f}, " +
                    f"loss: {logs['loss']:.4f}, " +
                    f"val_accuracy: {logs['val_accuracy']:.4f}, " +
                    f"val_loss: {logs['val_loss']:.4f}\n")

    def log_message(self, message):
        with open(self.file_name, 'a') as f:
            f.write(message + '\n')
