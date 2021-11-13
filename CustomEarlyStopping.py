# https://keras.io/guides/writing_your_own_callbacks/
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras



class CustomEarlyStopping(keras.callbacks.Callback):
    def __init__(self, patience=0, txtpath_to_check=None, model_save_path= None):
        super(CustomEarlyStopping, self).__init__()
        self._patience = patience
        self._txtpath_to_check = txtpath_to_check
        self._best_weights = None
        self._model_save_path = model_save_path

        return

    def on_train_begin(self, logs=None):
        self._wait=0
        self._stopped_epoch=0
        self._best_epoch = 0
        self._best=np.Inf

        with open(self._txtpath_to_check, "w") as f:
            f.write('O')
        return

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss") # loss -> training loss # val_loss -> validation loss
        #print("debug, what is logs.get('loss'), that is, current", current)
        txt=None
        if self._txtpath_to_check:
            # check txt file
            f = open(self._txtpath_to_check, "r")
            txt = f.readline()
            f.close()
        if txt == 'X' or txt=='X\n':
            self._stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self._best_weights)

            with open(self._txtpath_to_check, "w") as f:
                f.write('O')

            return
        if np.less(current, self._best):
            self._best=current
            self._wait = 0
            # Record the best weights if current results is better (less).
            self._best_weights = self.model.get_weights()
            self._best_epoch = epoch
            print("Saving model weights which made best performance so far at", epoch, " epoch")
            #print("Save best model weights.")
        else :
            self._wait +=1
            if self._wait >= self._patience:
                self._stopped_epoch = epoch
                self.model.stop_training=True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self._best_weights)
        return

    def on_train_end(self, logs=None):
        if self._stopped_epoch > 0:
            print("Epoch %05d: early stopping" %(self._stopped_epoch+1))

        # save model
        if self._model_save_path is not None:
            self.model.save(os.path.join(self._model_save_path, "model_" + str(self._best_epoch + 1) + ".h5"))


if __name__ == "__main__":
    print("test")
    with open('tes_kk.txt', 'w') as f:
        f.write("O")
