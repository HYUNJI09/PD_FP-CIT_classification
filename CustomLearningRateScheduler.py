# https://keras.io/guides/writing_your_own_callbacks/
# https://stackoverflow.com/questions/59737875/keras-change-learning-rate
# https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import keras

from Logger import Logger

class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule=None, window_size=10, log_save_path = None, patience=100, stzcr_threshold=0.05, decay_rate=0.05):
        super(CustomLearningRateScheduler, self).__init__()
        self._schedule = schedule
        self._window_size = window_size
        self._loss_window = []
        self._patience = patience
        self._logger = Logger(log_save_path)
        self._stzcr_threshold = stzcr_threshold
        self._decay_rate = decay_rate
        return

    def put_loss_in_window_queue(self, loss):
        self._loss_window.append(loss)
        if len(self._loss_window) > self._window_size:
            del self._loss_window[0]
        return

    def on_train_begin(self, logs=None):
        self._wait=0
        self._best=np.Inf
        # self._previous_loss = np.Inf
        self._previous_loss_list = [] #
        self._init_lr = float(keras.backend.eval(self.model.optimizer.lr))
        return

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = float(keras.backend.eval(self.model.optimizer.lr))
        print("\nEpoch %05d: Learning rate is %6f." % (epoch, lr))

        # Call schedule function to get the scheduled learning rate.
        if self._schedule is not None:
            scheduled_lr = self._schedule(epoch, lr)
            keras.backend.set_value(self.model.optimizer.learning_rate, scheduled_lr)
            print("\nEpoch %05d: Learning rate is %10f." % (epoch, scheduled_lr))
        else :
            if len(self._loss_window) >= self._window_size:
                _mean_subtracts = self.get_mean_subtraction_on_front_window(self._loss_window, self._window_size)
                stzcr_list = self.get_STZCR(_mean_subtracts, self._window_size)
                current_stzcr = stzcr_list[-1]
                print("current stzcr : ", current_stzcr)

                if current_stzcr > self._stzcr_threshold:
                    scheduled_lr = lr-lr*self._decay_rate # scheduled with STZCR by default
                    keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                    print("\nEpoch %05d: Learning rate decreased by perterbation (STZCR) : %10f."%(epoch, scheduled_lr))

                self._logger.log_msg(Epoch=epoch, lr_rate=lr, STZCR=current_stzcr)
        return

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")

        # log current loss into previous_loss_list
        self._previous_loss_list.append(current)

        self.put_loss_in_window_queue(current)
        lr = float(keras.backend.eval(self.model.optimizer.lr))

        if np.less(current, self._best):
            self._best=current
            self._wait = 0

        else :
            self._wait +=1
            if self._wait >= self._patience:
                scheduled_lr = lr - lr * self._decay_rate
                keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
                print("\nEpoch %05d: Learning rate decreased by wait : %10f." % (epoch, scheduled_lr))

        current_st_sqrt = self.get_sqrt_by_frame(self._previous_loss_list)[-1]
        print("current_st_sqrt : ", current_st_sqrt)
        if len(self._previous_loss_list) > self._window_size and current_st_sqrt<1e-3:
            # scheduled_lr = lr + lr * self._decay_rate
            scheduled_lr = self._init_lr
            keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
            print("\nEpoch %05d: Learning rate increased by fixation (st_sqrt) : %10f." % (epoch, scheduled_lr))

        #if np.greater(np.abs(current-self._previous_loss), 0.005):
        #    self._wait_for_fixation

        return
    def on_train_end(self, logs=None):
        self._logger.save_board_to_excel()
        return

    def get_mean_subtraction_on_front_window(self, signals, frame_size):
        frame_mean_list = []
        # convolution
        for ind, signal in enumerate(signals):
            current_ind = ind
            first_frame_ind = current_ind - (frame_size - 1)
            if first_frame_ind < 0:
                first_frame_ind = 0
            _mean = np.mean(signals[first_frame_ind:current_ind + 1])
            frame_mean_list.append(signal - _mean)
        return np.array(frame_mean_list)

    def get_STZCR(self, signals, frame_size):
        signals = np.array(signals)
        stzcr_list = []
        signals_t_1 = signals[:-1]
        signals_t_1 = np.insert(signals_t_1, 0, 0)

        # convolution
        for ind, signal in enumerate(signals):
            current_ind = ind
            first_frame_ind = current_ind - (frame_size - 1)
            if first_frame_ind < 0:
                first_frame_ind = 0
            _sum = np.sum(
                np.abs(signals_t_1[first_frame_ind:current_ind + 1] - signals[first_frame_ind:current_ind + 1]))
            stzcr_list.append(_sum)
        return np.array(stzcr_list)

    def get_sqrt_by_frame(self, signals, frame_size=10):
        frame_mean_list = []
        # convolution
        for ind, signal in enumerate(signals):
            current_ind = ind
            first_frame_ind = current_ind - (frame_size - 1)
            if first_frame_ind < 0:
                first_frame_ind = 0
            st_signals = np.array(signals[first_frame_ind:current_ind + 1])
            _mean = np.mean(st_signals)
            _sqrt = np.sum(np.sqrt(np.abs(st_signals - _mean)))
            frame_mean_list.append(_sqrt)
        return np.array(frame_mean_list)

def get_mean_subtraction_on_front_window(signals, frame_size):
    frame_mean_list = []
    # convolution
    for ind, signal in enumerate(signals):
        current_ind = ind
        first_frame_ind = current_ind-(frame_size-1)
        if first_frame_ind < 0:
            first_frame_ind = 0
        _mean = np.mean(signals[first_frame_ind:current_ind+1])
        frame_mean_list.append(signal-_mean)
    return np.array(frame_mean_list)

def sign(x):
    return (x>=0)+(-1)*(x<0)

# https://oasiz.tistory.com/175
def get_STZCR(signals, frame_size):
    signals = np.array(signals)
    stzcr_list = []
    signals_t_1 = signals[:-1]
    signals_t_1 = np.insert(signals_t_1, 0, 0)

    # convolution
    for ind, signal in enumerate(signals):
        current_ind = ind
        first_frame_ind = current_ind - (frame_size - 1)
        if first_frame_ind < 0:
            first_frame_ind = 0
        _sum = np.sum(np.abs(signals_t_1[first_frame_ind:current_ind+1] - signals[first_frame_ind:current_ind+1]))
        stzcr_list.append(_sum)
    return np.array(stzcr_list)

def get_sqrt_by_frame(signals, frame_size):
    frame_mean_list = []
    # convolution
    for ind, signal in enumerate(signals):
        current_ind = ind
        first_frame_ind = current_ind - (frame_size - 1)
        if first_frame_ind < 0:
            first_frame_ind = 0
        st_signals = np.array(signals[first_frame_ind:current_ind + 1])
        _mean = np.mean(st_signals)
        _sqrt = np.sum(np.sqrt(np.abs(st_signals-_mean)))
        frame_mean_list.append(_sqrt)
    return np.array(frame_mean_list)

if __name__ == "__main__":
    print('test')
    test_data_path = os.path.join(r'C:\Users\hkang\PycharmProjects\DynamicPETModeling\lstm_experimental_results_200814_early2', r'trend_log_cv1.xlsx')
    colname = 'test_loss_list'

    pd_df = pd.read_excel(test_data_path)
    test_loss = pd_df[colname].tolist()
    print(test_loss)
    # test_loss = test_loss[:1400]
    #
    frame_mean_list = get_mean_subtraction_on_front_window(test_loss, frame_size=10)
    stzcr_list = get_STZCR(frame_mean_list, frame_size=10)
    st_sqrt_list = get_sqrt_by_frame(test_loss, frame_size=10)
    print("st_sqrt_list", st_sqrt_list)
    # target_ind =0
    # for ind, _stzcr in enumerate(stzcr_list):
    #     if _stzcr > 0.05:
    #         target_ind = ind
    #         break
    # test_loss = test_loss[:target_ind]
    # frame_mean_list = frame_mean_list[:target_ind]
    # stzcr_list = stzcr_list[:target_ind]

    final_stzcr = stzcr_list[-1]
    print("final_stzcr", final_stzcr, np.max(stzcr_list))

    x = np.arange(len(test_loss))
    plt.plot(x, test_loss, color='b')
    plt.plot(x, frame_mean_list, color='r')
    plt.plot(x, stzcr_list, color='g')
    plt.plot(x, st_sqrt_list, color='k')
    plt.axhline(0, color='k')
    plt.show()

