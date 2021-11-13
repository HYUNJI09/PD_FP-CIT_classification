# https://keras.io/guides/writing_your_own_callbacks/

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class LossLoggerScheduler(keras.callbacks.Callback):
    def __init__(self, logger):
        super(LossLoggerScheduler, self).__init__()
        self._logger = logger
        return

    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        # print("keys", keys)
        train_loss = logs['loss']
        val_loss = logs['val_loss']
        try:
            train_acc = logs['acc']
            val_acc = logs['val_acc']
        except KeyError as ke:
            train_acc = logs['accuracy']
            val_acc = logs['val_accuracy']
        self._logger.log_msg(num_epoch=str(epoch+1), train_loss=train_loss, train_acc=train_acc,
        val_loss=val_loss, val_acc=val_acc)
        self._logger.save_board_to_excel()

        total_train_loss = self._logger.board['train_loss']
        total_train_acc = self._logger.board['train_acc']
        total_val_loss = self._logger.board['val_loss']
        total_val_acc = self._logger.board['val_acc']

        self.plot_trend(train_loss=total_train_loss, val_loss=total_val_loss,
        save_path=os.path.join(self._logger.log_save_path, "logs.png"),
        train_acc=total_train_acc, val_acc=total_val_acc)
        return

    def plot_trend(self, train_loss, val_loss, save_path, train_acc=None, val_acc=None):
        # for item in trend_dict.items(): # key value pair i.g. "train_acc": 90.0
        #     plt.plot(item[1], label=item[0])
        # plt.legend('upper right')
        # plt.savefig(save_path)
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(train_loss, 'y', label='train loss')
        loss_ax.plot(val_loss, 'r', label='val loss')
        loss_ax.legend(loc='upper left')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        if train_acc is not None:
            acc_ax.plot(train_acc, 'b', label='train acc')
            acc_ax.plot(val_acc, 'g', label='val acc')
            acc_ax.set_ylabel('accuray')
            acc_ax.legend(loc='lower left')

        plt.savefig(save_path)
        plt.cla()
        plt.clf()
        plt.close()
        return
