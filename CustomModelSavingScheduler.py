# https://keras.io/guides/writing_your_own_callbacks/
import os
import tensorflow as tf
from tensorflow import keras

class CustomModelSavingScheduler(keras.callbacks.Callback):
    def __init__(self, model_save_path, saving_step=None):
        super(CustomModelSavingScheduler, self).__init__()
        self._model_save_path = model_save_path
        self._saving_step=saving_step
        return

    def on_train_begin(self, logs=None):
        self._best_accuracy=0
        return

    def on_epoch_end(self, epoch, logs=None):
        #print(list(logs.keys()))
        if self._saving_step is not None:
            if (epoch+1)%self._saving_step==0:
                self.model.save(os.path.join(self._model_save_path, "model_"+str(epoch+1)+".h5"))
        else :
            try:
                current = logs['val_acc']
            except KeyError as ke:
                current = logs['val_accuracy']
            if current > self._best_accuracy:
                print("[!] renew last best accuracy", self._best_accuracy)
                self._best_accuracy = current
                self.model.save(os.path.join(self._model_save_path, "model_"+str(epoch+1)+".h5"))
        return
