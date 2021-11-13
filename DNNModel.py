import os
import sys
import operator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf

from Resnet3D import Resnet3D
from conv3x3_3D import CNN3D
from DenseNet3D import DenseNet3D
from Inception3DModel import Inception3D
from model import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

# keras modules
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils import print_summary
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# keras modules to design a model_name
from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add, concatenate
from keras.layers import Dense, Concatenate, Lambda, Add
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model
from keras.optimizers import Adam

# customized modules
from img_utils import thresholding_matrix
# from model_name import create_3d_cnn_model
from preprocess import StandardScalingData
from ImageDataIO import ImageDataIO
from set_directory import set_directory
from data_load_utils import save_pred_label
from custom_exception import *
from Interpolation3D import ImageInterpolator
from retrieve import make_col_list_from_excel
from img_utils import draw_img_on_grid
from npy_utils import npy_mat_saver


class DNNModel():
    # Model parameter & experimental configuration
    def __init__(self, num_classes, lr_rate=0.00005, num_epochs=100, batch_size=64):
        # hyper parameters
        self._num_classes = num_classes
        self._lr_rate = lr_rate
        self._num_epochs = num_epochs

        self._train_batch_size = batch_size
        self._test_batch_size = batch_size

        self._datas = None
        self._labels = None
        self._data_filenames = None
        self._label_name = None
        self._class_name = None

        self._model = None
        self._scaler = None

    def clear_model(self):
        if self._model is not None:
            del self._model
        return

    def LoadData(self, datas=None, labels=None, data_filenames=None, label_name=None, class_name=None):
        """

        :param datas: this codes expect the inputs to be arranged features (N, F)
        :param labels:
        :param data_filenames:
        :param label_name:
        :param class_name:
        :return:
        """
        if datas is not None:
            self._datas = datas  # It is not decided whether datas will be used for training or testing

        if labels is not None:
            self._labels = labels

        if label_name is not None:
            self._label_name = label_name

        if data_filenames is not None:
            self._data_filenames = data_filenames

        if class_name is not None:
            self._class_name = class_name

        return

        # CV for model selection
        def train_eval_CV(self, num_cv):
            self.clear_model()
            kf = KFold(n_splits=num_cv, random_state=1)

            train_acc_list = []
            train_f1_score_list = []
            val_acc_list = []
            val_f1_score_list = []

            for ind, (train_index, test_index) in enumerate(kf.split(self._datas, self._labels)):
                # for ind in range(num_cv):
                print("#CV:", ind)
                # train_index = pd_cv_index_train["train_"+str(ind)].tolist()
                # test_index = pd_cv_index_test["test_" + str(ind)].tolist()
                X_train = self._datas[train_index]
                y_train = self._labels[train_index]
                X_test = self._datas[test_index]
                y_test = self._labels[test_index]
                print("X_train", X_train.shape)
                print("y_train", y_train.shape)
                print("X_test", X_test.shape)
                print("y_test", y_test.shape)

                train_acc, train_f1_score, val_acc, val_f1_score = self.train_eval(X_train, y_train, X_test, y_test)

                train_acc_list.append(val_acc)
                train_f1_score_list.append(train_f1_score)
                val_acc_list.append(val_acc)
                val_f1_score_list.append(val_f1_score)

            print("Final Train Acc:", np.mean(train_acc_list))
            print("Final Train F1-Score:", np.mean(train_f1_score_list))
            print("Final Val Acc:", np.mean(val_acc_list))
            print("Final Val F1-Score:", np.mean(val_f1_score_list))
            return train_acc, train_f1_score, val_acc, val_f1_score

        # train and evaluate model_name : [model_name validation]
        def train_eval(self,  X_train, y_train, X_test, y_test, X_train_filename=None, X_test_filename=None, output_save_dir=None):
            self.clear_model()

            tmp_datas = self._datas
            tmp_labels = self._labels
            tmp_data_filenames = self._data_filenames
            tmp_label_name = self._label_name
            tmp_class_name = self._class_name

            self.LoadData(datas=X_train, labels=y_train, data_filenames=X_train_filename, label_name=self._label_name,
                          class_name=self._class_name)
            train_accuracy, train_loss = self.train()

            self.LoadData(datas=X_test, labels=y_test, data_filenames=X_test_filename, label_name=self._label_name,
                          class_name=self._class_name)
            val_accuracy, val_loss = self.test(output_save_dir)

            self.LoadData(datas=tmp_datas, labels=tmp_labels, data_filenames=tmp_data_filenames,
                          label_name=tmp_label_name,
                          class_name=tmp_class_name)
            return train_accuracy, train_loss, val_accuracy, val_loss

    def train(self, output_save_dir=None, print_step=10):
        print("[!] DNN is prepared for training")

        # preprocessing : StandardScalar
        # self._scaler, scaled_X_train = StandardScalingData(X_train, save_path=None, keep_dim=True, train=True, scaler=None)

        # categorical_train_label = to_categorical(y_train, self._num_classes)
        self._input_shape = (self._datas.shape[1], )
        model = load_simple_DNN(lr_rate=self._lr_rate, num_classes=self._num_classes, input_shape=self._input_shape)
        history = model.fit(_x_train, _bin_y_train_categorical_label, epochs=num_epochs, verbose=1,
                            validation_data=(_x_val, _bin_y_val_categorical_label),
                            callbacks=[
                                # CustomModelSavingScheduler(model_save_path),
                                LossLoggerScheduler(customlogger),
                                CustomEarlyStopping(patience=500, txtpath_to_check=acync_stopping_txtpath,
                                                    model_save_path=model_save_path),
                                # ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10),
                                CustomLearningRateScheduler(window_size=10, log_save_path=cv_lr_log_save_path,
                                                            stzcr_threshold=0.05)
                            ])

        train_predict = self._model.predict(self._datas)

        train_accuracy = accuracy_score(train_predict, self._labels)
        train_f1_score = f1_score(train_predict, self._labels)
        print("Training Accuracy: {0:.6f}".format(train_accuracy))
        print("Training F1_score: {0:.6f}".format(train_f1_score))

        return train_accuracy, train_loss

    def test(self, output_save_dir):

        # preprocessing : StandardScalar
        # _, scaled_X_test = StandardScalingData(X_test, save_path=None, keep_dim=True, train=False, scaler=self._scaler)

        # categorical_val_label = to_categorical(y_test, self._num_classes)



        if output_save_dir:

            # logging the conf_m
            with open(output_save_dir + "\\" + "best_model_performance_report_conf_m.txt", "w") as f:
                f.write("Accuracy : " + str(val_accuracy) + '\n')
                for row in conf_m:
                    f.write("%s\n" % row)

            target_names = ['BAPL1', 'BAPL3']

            result = classification_report(self._labels, val_predict, target_names=target_names)
            print("Test Phase result")
            print(result)
            with open(output_save_dir + "\\" + "best_model_performance_report_conf_m.txt", "a") as f:
                f.write("%s\n" % result)

        return val_accuracy, val_loss


    # train and evaluate model_name : [model_name validation]
    def train_eval_add_info_ver(self, train_data, train_label, val_data, val_label, experiment_name):


        if not os.path.isdir(os.path.join(self._root_dir, self._model_save_dir, experiment_name)):
            os.mkdir(os.path.join(self._root_dir, self._model_save_dir, experiment_name))
        if not os.path.isdir(os.path.join(self._root_dir, self._result_save_dir, experiment_name)):
            os.mkdir(os.path.join(self._root_dir, self._result_save_dir, experiment_name))

        conv_train_data = self._preprocess(train_data[0], isTrain=True, experiment_name=experiment_name+"_conv")
        conv_val_data = self._preprocess(val_data[0], isTrain=False, experiment_name=experiment_name+"_conv")
        series_train_data = self._preprocess(train_data[1], isTrain=True, experiment_name=experiment_name + "_series")
        series_val_data = self._preprocess(val_data[1], isTrain=False, experiment_name=experiment_name + "_series")

        train_data = (conv_train_data, series_train_data)
        val_data = (conv_val_data, series_val_data)

        categorical_train_label = to_categorical(train_label, self._num_classes)
        categorical_val_label = to_categorical(val_label, self._num_classes)

        # datagen_train = ImageDataGenerator()
        # datagen_test = ImageDataGenerator()
        #
        # # ImageDataGenerator for Outer Loop
        # generator_train = datagen_train.flow(x=train_data, y=categorical_train_label, batch_size=self._train_batch_size, shuffle=True)
        # generator_test = datagen_test.flow(x=val_data, y=categorical_eval_label, batch_size=self._test_batch_size, shuffle=False)
        #
        save_filepath = os.path.join(self._root_dir, self._result_save_dir, experiment_name,
                                     "final_selected_model_structure.txt")

        def _model_summary_log(line, save_filepath):
            with open(save_filepath, "a") as f:
                f.write(line + "\n")
            return

        print_summary(self._model, line_length=None, positions=None,
                      print_fn=lambda line: _model_summary_log(line, save_filepath))
        # model_plot_savepath = os.path.join(self._root_dir, self._result_save_dir,
        #                              "final_selected_model_structure.png")
        # plot_model(self._model, model_plot_savepath)
        # Early Stopping
        #earlystopping = EarlyStopping(monitor='val_loss', patience=10)

        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_label), train_label)
        class_weight_dict = dict(enumerate(class_weights))
        print("class_weight_dict", class_weight_dict) # {0: 4.666666666666667, 1: 0.56}
        # tb_hist = keras.callbacks.TensorBoard(log_dir='./results/outer_loop/CV'+str(ind+1)+'graph', histogram_freq=0, write_graph=True, write_images=True)
        steps = int(len(train_label) / self._train_batch_size)
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        for epoch in tqdm(range(self._num_epochs)):
            print("# Epoch:", epoch)
            for _ in range(steps):
                # Select a random batch of images
                idx = np.random.randint(0, len(train_data), self._train_batch_size)
                imgs, features, labels = train_data[0][idx], train_data[1][idx], categorical_train_label[idx]
                #print("labels shape", labels.shape) # (64,)
                # Train & Test the model_name
                train_loss = self._model.train_on_batch(x=[imgs, features], y=labels, class_weight=class_weight_dict)
                test_loss = self._model.test_on_batch(x=[val_data[0], val_data[1]], y=categorical_val_label)
            train_loss_list.append(train_loss[0])
            train_acc_list.append(train_loss[1])
            test_loss_list.append(test_loss[0])
            test_acc_list.append(test_loss[1])

        # history = self._model.fit(train_data, categorical_train_label, batch_size=self._train_batch_size,
        #                           epochs=self._num_epochs, shuffle=True,
        #                           validation_data=(val_data, categorical_val_label), callbacks=[earlystopping])

        # train_acc_list = history.history['acc']
        # train_loss_list = history.history['loss']
        # val_acc_list = history.history['val_acc']
        # val_loss_list = history.history['val_loss']

        trend_dict = dict()
        trend_dict["train_acc_list"] = train_acc_list
        trend_dict["train_loss_list"] = train_loss_list
        trend_dict["test_acc_list"] = test_acc_list
        trend_dict["test_loss_list"] = test_loss_list
        trend_df = pd.DataFrame(trend_dict)
        trend_df.to_excel(os.path.join(self._root_dir, self._model_save_dir, experiment_name, "trend_log.xlsx"))

        # save history log for checking trend info later
        trend_save_path = os.path.join(self._root_dir, self._result_save_dir, experiment_name, "trend_figure.png")
        self._plot_trend(trend_dict, save_path=trend_save_path)

        # accuracy = history.history['val_acc'][-1]
        # val_loss = history.history['val_loss'][-1]
        print("Accuracy: {0:.4%}".format(test_acc_list[-1]))
        print("Val Loss: {0:.4}".format(test_loss_list[-1]))

        return train_acc_list[-1], train_loss_list[-1], test_acc_list[-1], test_loss_list[-1]

    # output of CV is probably best appoximate performance for selected model_name
    def train_eval_CV(self, num_cv=10):
        model_acc = []
        model_loss = []

        #default_cv_file_path = os.path.join(self._root_dir, self._result_save_dir, "cv_file.xlsx")
        default_cv_file_path = None
        #default_cv_file_path = True
        #print("debug", self._label)

        #pd_cv_index_train, pd_cv_index_test = self._create_cv_file(self._3D_data, self._label, num_k = num_cv, save_path=default_cv_file_path)
        kf = KFold(n_splits=num_cv, random_state=1)

        # cv_index_train = dict()
        # cv_index_test = dict()
        # sss_inner.get_n_splits(data, label)
        # for ind, (train_index, test_index) in enumerate(sss_inner.split(data, label)):
        for ind, (train_index, test_index) in enumerate(kf.split(self._3D_data, self._label)):
        #for ind in range(num_cv):
            print("#CV:", ind)
            # train_index = pd_cv_index_train["train_"+str(ind)].tolist()
            # test_index = pd_cv_index_test["test_" + str(ind)].tolist()
            X_train = self._3D_data[train_index]
            y_train = self._label[train_index]
            X_test = self._3D_data[test_index]
            y_test = self._label[test_index]
            print("X_train", X_train.shape)
            print("y_train", y_train.shape)
            print("X_test", X_test.shape)
            print("y_test", y_test.shape)

            if self._train_batch_size > len(X_train):
                self._train_batch_size = len(X_train)
            if self._test_batch_size > len(X_test):
                self._test_batch_size = len(X_test)

            # for additional clinical information
            # add_X_train = self._features_sorted_by_filename[train_index]
            # X_train = (X_train, add_X_train)
            # add_X_test = self._features_sorted_by_filename[test_index]
            # X_test = (X_test, add_X_test)

            test_filename = self._3D_data_filename[test_index]

            # name of model to save
            # save_model_name = self._experiment_name + "_" + "_".join((str(_dim) for _dim in self._input_shape))  + ".h5"
            save_model_name = self._root_dir + "_" + "_".join((str(_dim) for _dim in self._input_shape)) + ".h5"
            _, _, val_acc, val_loss = self.train_eval(X_train, y_train, X_test, y_test,
                                                      experiment_name="experiment_CV_"+str(ind), save_model_name=save_model_name)
            preds = self.test(X_test, y_test, experiment_name="experiment_CV_"+str(ind), filename_list=test_filename)
            #_, _, val_acc, val_loss = self.train_eval_add_info_ver(X_train, y_train, X_test, y_test, experiment_name="experiment_CV_" + str(ind))
            #self.test_add_info_ver(X_test, y_test, experiment_name="experiment_CV_" + str(ind), filename_list=test_filename)

            print("[!] preds", preds.shape) # (89, 2), softmax_output as a last layer of model_name
            # CAM_save_dir = os.path.join(self._root_dir, self._result_save_dir, "CAM_log_"+str(ind))
            # if not os.path.isdir(CAM_save_dir):
            #     os.mkdir(CAM_save_dir)

            # self._visualize_3D_CAM(X_test, y_test, preds, test_filename, CAM_save_dir, featuremap_name="mixed0",
            #                        resize_size = (64,95,79), experiment_name_for_preprocess="experiment_CV_"+str(ind))
            # self._visualize_3D_CAM(X_test, y_test, preds, test_filename, CAM_save_dir,
            #                        featuremap_name="conv5_block16_concat",
            #                        resize_size=(64, 95, 79), experiment_name_for_preprocess="experiment_CV_" + str(ind))
            model_acc.append(val_acc)
            model_loss.append(val_loss)

        print("Final Acc:", np.mean(model_acc))
        print("Final Loss:", np.mean(model_loss))


        return np.mean(model_acc), np.mean(model_loss)

    # output of CV is probably best approximate performance and optimized model_name
    def train_eval_NCV(self, train_data, train_label, val_data, val_label, num_cv=10, cv_file=None, model_save_dir=None):
        model_acc = None
        model_loss = None
        best_model = None

        return model_acc, model_loss, best_model

    # save model_name
    # def _save_model(self, filename):
    #     save_path = os.path.join(self._root_dir, self._model_save_dir, filename)
    #     self._model.save(save_path)
    #     return

    def saveModel(self, experiment_name=None, save_model_name = None):
        if save_model_name is None:
            save_model_name = "model_name.h5"

        if experiment_name :
            self._model.save(os.path.join(self._root_dir, self._model_save_dir, experiment_name, save_model_name))
        else :
            self._model.save(os.path.join(self._root_dir, self._model_save_dir, save_model_name))

    # test model_name with label data, because this phase is literally 'test'
    def test(self, val_data, val_label, experiment_name, filename_list, preprocess_save_path=None):

        val_data = self._preprocess(val_data, isTrain=False, experiment_name=experiment_name, save_path=preprocess_save_path)
        categorical_test_label = to_categorical(val_label, self._num_classes)

        # datagen_test = ImageDataGenerator()
        # generator_test = datagen_test.flow(x=self._3D_data, y=categorical_test_label, batch_size=self._test_batch_size, shuffle=False)

        preds = self._model.predict(x=val_data, verbose=1) # (N, num_classes)
        print("preds", np.array(preds).shape) # (89, 2)

        # pred_proba = np.array(pred).max(axis=1)  # (N, )
        pred_ind_list = np.array(preds).argmax(axis=1)  # (N, )
        label_ind_list = np.array(categorical_test_label).argmax(axis=1)  # (N, )

        if not os.path.isdir(os.path.join(self._root_dir, self._result_save_dir, experiment_name)):
            os.mkdir(os.path.join(self._root_dir, self._result_save_dir, experiment_name))
        # sample_based_analysis result
        sample_based_analysis_save_filename = os.path.join(self._root_dir, self._result_save_dir, experiment_name,
                                                           "sample_based_analysis.xlsx")
        save_pred_label(categorical_test_label, preds, save_filepath=sample_based_analysis_save_filename,
                        onehot_label = True, filename_list = filename_list) # y_test_ : (N, Num_classes)

        conf_m = confusion_matrix(label_ind_list, pred_ind_list)
        print("CV #", "confusion matrix")
        print(conf_m)

        accuracy = accuracy_score(y_true=label_ind_list, y_pred=pred_ind_list)

        # logging the conf_m
        result_save_filename = os.path.join(self._root_dir, self._result_save_dir, experiment_name,
                                            "performance_report.txt")
        with open(result_save_filename, "w") as f:
            f.write("Accuracy : " + str(accuracy) + '\n')
            for row in conf_m:
                f.write("%s\n" % row)

        target_names = np.unique(self._label_name)
        result = classification_report(label_ind_list, pred_ind_list, target_names=target_names)
        print("Test Phase result")
        print(result)
        with open(result_save_filename, "a") as f:
            f.write("%s\n" % result)

        return preds

    # test model_name with label data
    def test_add_info_ver(self, val_data, val_label, filename_list, experiment_name, load_path=None):
        if load_path is not None:
            self.loadModel(input_shape=(64, 32, 32, 1), load_path=load_path)
        conv_test_data = self._preprocess(val_data[0], isTrain=False, experiment_name=experiment_name+"_conv")
        feature_test_data = self._preprocess(val_data[1], isTrain=False, experiment_name=experiment_name+"_series")
        categorical_test_label = to_categorical(val_label, self._num_classes)

        # datagen_test = ImageDataGenerator()
        # generator_test = datagen_test.flow(x=self._3D_data, y=categorical_test_label, batch_size=self._test_batch_size, shuffle=False)

        #preds = self._model.predict(x=self._3D_data, verbose=1)  # (N, num_classes)
        preds = self._model.predict_on_batch(x=[conv_test_data, feature_test_data])
        print("preds", np.array(preds).shape)

        # pred_proba = np.array(pred).max(axis=1)  # (N, )
        pred_ind_list = np.array(preds).argmax(axis=1)  # (N, )
        label_ind_list = np.array(categorical_test_label).argmax(axis=1)  # (N, )

        # sample_based_analysis result
        sample_based_analysis_save_filename = os.path.join(self._root_dir, self._result_save_dir, experiment_name,
                                                           "sample_based_analysis.xlsx")
        save_pred_label(label_ind_list, pred_ind_list, save_filepath=sample_based_analysis_save_filename,
                        onehot_label=False, filename_list=filename_list)  # y_test_ : (N, Num_classes)

        conf_m = confusion_matrix(label_ind_list, pred_ind_list)
        print("CV #", "confusion matrix")
        print(conf_m)

        accuracy = accuracy_score(y_true=label_ind_list, y_pred=pred_ind_list)

        # logging the conf_m
        result_save_filename = os.path.join(self._root_dir, self._result_save_dir, experiment_name, "performance_report.txt")
        with open(result_save_filename, "w") as f:
            f.write("Accuracy : " + str(accuracy) + '\n')
            for row in conf_m:
                f.write("%s\n" % row)

        #target_names = np.unique(self._label_name)
        target_names = self._class_name
        result = classification_report(label_ind_list, pred_ind_list, target_names=target_names)
        print("Test Phase result")
        print(result)
        with open(result_save_filename, "a") as f:
            f.write("%s\n" % result)

        return

    # predict data without label, because this phase is literally 'predict' and can be used for deployment
    def predict(self, val_data, experiment_name, load_path=None, preprocess_save_path=None):
        if load_path is not None and self._model is None:
            self.loadModel(input_shape=(64, 64, 64, 1), load_path=load_path)
        val_data = self._preprocess(val_data, isTrain=False, experiment_name=experiment_name, save_path=preprocess_save_path)

        # datagen_test = ImageDataGenerator()
        # generator_test = datagen_test.flow(x=self._3D_data, y=categorical_test_label, batch_size=self._test_batch_size, shuffle=False)

        preds = self._model.predict(x=val_data, verbose=1) # (N, num_classes)
        print("preds", np.array(preds).shape)

        return preds



    def visualize_3D_CAM_from_data(self, input_data, extension, featuremap_name, preprocess_save_path,
                                   reload_data = True, data_dir_child="labeled", reload_model_path=None, heatmap_resize_size=None):
        """
        :param input_data: when reload_data is True, input_data is a directory. if not, input_data is a tuple of matrice like (X_data, label, filename)
        :param extension:
        :param CAM_save_dir:
        :param featuremap_name:
        :param preprocess_save_path: to load a step of preprocess
        :param reload_data: if True, this module update self._3D_data from the "input_data" which is labeled. if False, input_data is matrice
        :param data_dir_child:
        :param reload_model_path: this parameter is used to load model_name when self._model object doesn't exist and user didn't call loadModel() function before this function,

        :return: only predictions to provide service (that's why this function doen't requare 'label' parameter.
        """

        if reload_data and data_dir_child is "labeled":
            conv3dmodel.load3DImage(input_data, extension=extension, data_dir_child="labeled", dataDim=dataDim,
                                instanceIsOneFile=True, view=view)
            X_data = self._3D_data
            label = self._label
            filename = self._3D_data_filename
        elif reload_data and data_dir_child is None:
            conv3dmodel.load3DImage(input_data, extension=extension, dataDim=dataDim,
                                    instanceIsOneFile=True, view=view)
            X_data = self._3D_data
            label = self._label
            filename = self._3D_data_filename
        elif reload_data is False:
            CustomException("[!] not considered yet! ")
            X_data = input_data[0]
            label = input_data[1]
            filename = input_data[2]
        #preds = self.test(X_data, label, experiment_name="experiment_TV", filename_list=filename, preprocess_save_path=preprocess_save_path)
        preds = self.predict(X_data, experiment_name=None, load_path=reload_model_path, preprocess_save_path=preprocess_save_path)
        CAM_save_dir = os.path.join(self._root_dir, self._result_save_dir, "CAM")
        if not os.path.isdir(CAM_save_dir):
            os.mkdir(CAM_save_dir)
        self._visualize_3D_CAM(X_data, label, preds, filename, CAM_save_dir, featuremap_name=featuremap_name,
                               resize_size=heatmap_resize_size, preprocess_save_path=preprocess_save_path)

        return

    """
    # show CAM
    - only convolutional feature space
    - with clinical variables
    """
    # save 3D CAM for train_eval phase
    def _visualize_3D_CAM(self, imgs_list, imgs_label, _preds, imgs_filename, CAM_save_dir, featuremap_name,
                         resize_size=None, experiment_name_for_preprocess=None, preprocess_save_path=None):

        print("[!] visualizing 3D CAM")
        print("imgs_list", imgs_list.shape)
        print("imgs_label", imgs_label.shape)
        print("_preds", _preds.shape)

        # preprocessing
        if experiment_name_for_preprocess :
            preprocessed_imgs_list = self._preprocess(imgs_list, isTrain=False, experiment_name=experiment_name_for_preprocess)
        elif preprocess_save_path:
            preprocessed_imgs_list = self._preprocess(imgs_list, isTrain=False,
                                                      save_path=preprocess_save_path)
        else :
            preprocessed_imgs_list = imgs_list

        # def get_output_layer(model, layer_name):
        #     # # get the symbolic outputs of each "key" layer (we gave them unique names).
        #     # layer_dict = dict([(layer.name, layer) for layer in model_name.layers])
        #     # # print("layer dict", layer_dict)
        #     # layer = layer_dict[layer_name]
        #     # return layer
        #     for layer in model.layers:
        #         if layer.name == layer_name:
        #             return layer

        #get_output = K.function([self._model.layers[0].input], [final_conv_layer.output, self._model.layers[-1].output])
        class_heatmap_dict = dict()
        for ind2, (img, label, filename) in enumerate(zip(preprocessed_imgs_list, imgs_label, imgs_filename)):

            original_img = np.array([img])
            #print("original_img shape", original_img.shape) # (1, 64, 32, 32, 1)
            #print("original_img min max", original_img.min(), original_img.max())

            #[conv_outputs, predictions] = get_output([original_img])
            #print("conv_outputs", np.array(conv_outputs).shape)  # (1, 16, 8, 8, 128), (1, 4, 4, 4, 256)
            #print("predictions", np.array(predictions).shape)  # (1, 3)
            # print("from get_output()", predictions) # [[0. 1.]] same with softmax outputs
            # print("from _preds", _preds[ind2]) # [0.547228   0.45277202], outputs of keras models' predict() function
            # print("filename", filename)

            #conv_outputs = conv_outputs[0, :, :, :, :]  # (D, H, W, C) ; (16, 8, 8, 128)


            origin_heatmap_c = []
            for _c_ind in range(self._num_classes):  # iteration on each class
                # Create the class activation map.
                # cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3]) # H, W - 14,14
                #cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2+1])

                _class_output = self._model.output[:, _c_ind] # (N, num_classes)

                # Get the last layer's feature map
                final_conv_layer = self.get_output_layer(self._model, featuremap_name) # (1, D, H, W, C)

                # Get the last layer's input weights to the softmax.
                grads = K.gradients(_class_output, final_conv_layer.output)[0]

                # 특성 맵 채널별 그래디언트 평균 값이 담긴 (512,) 크기의 벡터
                pooled_grads = K.mean(grads, axis=(0, 1, 2, 3)) # (C, )
                iterate = K.function([self._model.input], [pooled_grads, final_conv_layer.output[0]])
                pooled_grads_value, conv_layer_output_value = iterate([original_img])
                for i, w in enumerate(range(len(pooled_grads_value))):  # 512 iter
                #for i, w in enumerate(std_last_class_weights[:, _c_ind]):
                    # cam += w * conv_outputs[i, :, :] # sumarize 16,16 on 512 iter with 512 weight for 1 label class
                    #cam += np.abs(w) * np.abs(conv_outputs[:, :, :, i])  # sumarize 16,16 on 512 iter with 512 weight for 1 label class
                    # normalize each of feature map from last convolutional feature bank
                    #conv_feature_map = conv_outputs[:, :, :, i]

                    #cam += w * normed_feature_map[:,:,:, i]
                    conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]
                    # cam += np.abs(w) * normed_feature_map[:,:,:, i]
                    #cam += np.abs(w) * conv_outputs[:, :, :, i]
                    #cam += w * conv_outputs[:, :, :, i]
                heatmap = np.mean(conv_layer_output_value, axis=-1)

                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                origin_heatmap_c.append(heatmap)

            origin_heatmap_c = np.array(origin_heatmap_c)
            print("origin_heatmap_c", origin_heatmap_c.shape) # (3, 4, 4, 4)
            #min_val = min(origin_heatmap_c[0].min(), origin_heatmap_c[1].min(), origin_heatmap_c[2].min())
            # min_val = min([origin_heatmap_c[_c_ind].min() for _c_ind in range(self._num_classes)])

            # most_wide_range = max((origin_heatmap_c[0].max() - origin_heatmap_c[0].min()),
            #                       (origin_heatmap_c[1].max() - origin_heatmap_c[1].min()),
            #                       (origin_heatmap_c[2].max() - origin_heatmap_c[2].min()))
            # most_wide_range = max([origin_heatmap_c[_c_ind].max() - origin_heatmap_c[_c_ind].min() for _c_ind in range(self._num_classes)])
            #most_wide_range = max([_c_heatmap.max() - _c_heatmap.min() for _c_heatmap in origin_heatmap_c])
            #max_val = max([origin_heatmap_c[_c_ind].max() for _c_ind in range(self._num_classes)])
            #most_wide_range = max_val - min_val

            #min_val = origin_heatmap_c.min()
            #most_wide_range = origin_heatmap_c.max() - min_val
            norm_heatmap = []
            # norm each heatmap to [0, 1]
            for h_ind, hm in enumerate(origin_heatmap_c):

                #tmp_hm = np.copy(hm)
                # min_val_regional_focus = tmp_hm.min()
                # range_val_regional_focus = tmp_hm.max() - tmp_hm.min()
                # tmp_hm -= min_val_regional_focus
                # tmp_hm = tmp_hm / range_val_regional_focus

                # min_val = hm.min()
                # most_wide_range = hm.max()-hm.min()
                # tmp_hm -= min_val
                # tmp_hm /= most_wide_range

                # print("heatmap histogram")
                # plt.hist(hm.ravel(), bins=256, fc='k', ec='k')
                # plt.show()

                #hm = cv2.resize(hm, (64, 64))
                if resize_size is not None:
                    imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=resize_size)
                    hm = imgip.interpolateImage(hm)
                    #tmp_hm = imgip.interpolateImage(tmp_hm)

                # min_val_after_resize = tmp_hm.min()
                # most_wide_range_after_resize = tmp_hm.max() - tmp_hm.min()
                # tmp_hm -= min_val_after_resize
                # tmp_hm /= most_wide_range_after_resize

                #print("hm shape", np.array(hm).shape) # (64, 97, 75)
                #print("img shape", np.array(img).shape) #  (64, 97, 75, 1)
                # heatmap = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
                # heatmap[np.where(hm < 0.4)] = 0  # thresholding
                # norm_heatmap.append(heatmap)

                #heatmap = np.array([cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm_d), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) for hm_d in tmp_hm]) # why...?? do i have to do 1-hm?
                #print("[!] debug heatmap shape", heatmap.shape)
                # for d_ind, heatmap_d in enumerate(heatmap):
                #     heatmap_d[np.where(hm[d_ind] < 0.4)] = 0




                # creating mask

                # threshold = tmp_hm.min() + (tmp_hm.max() - tmp_hm.min()) * 0.7 # (D, H, W)
                # bool_mask = tmp_hm > threshold  # region to show
                # int_mask = bool_mask.astype(np.uint8)
                # int_color_mask = np.array([int_mask, int_mask, int_mask])  # (3, D, H, W)
                # int_color_mask = np.transpose(int_color_mask, axes=(1, 2, 3, 0))
                #
                # heatmap = heatmap * int_color_mask

                #heatmap[np.where(tmp_hm<0.4)] = 0
                # print("heatmap index", h_ind)
                # print("heatmap shape", heatmap.shape)  # heatmap shape (64, 97, 75, 3)
                # print("heatmap min, max", heatmap.min(), heatmap.max()) # heatmap min, max 0 255
                norm_heatmap.append(hm)
                #norm_heatmap.append(tmp_hm)

                # visualization
                # idio = ImageDataIO(extension="nii", is2D=False, view="axial")
                # idio.show_one_img(hm, cmap=plt.get_cmap('jet'))
                # idio.show_one_img(heatmap, cmap=plt.get_cmap('jet'))
                # plt.imshow(heatmap, cmap=plt.get_cmap('jet'))
                # plt.colorbar(ticks=[0, 63, 127, 255], orientation='vertical')
                # plt.show()

                # plt.imshow(hm)
                # plt.show()
                # heatmap[np.where(cam < 0.2)] = 0
                # img = heatmap*0.5 + original_img

            norm_heatmap = np.array(norm_heatmap)

            # for elem_hm in norm_heatmap:
            #     elem_hm[0, 0, 0, :] = 255
            #     elem_hm[1, 0, 0, :] = 0

            for _c_ind in range(self._num_classes):
                try :
                    class_heatmap_dict[str(_c_ind)].append(norm_heatmap[_c_ind])
                except KeyError as ke:
                    class_heatmap_dict[str(_c_ind)]=[]
                    class_heatmap_dict[str(_c_ind)].append(norm_heatmap[_c_ind])

            # for dict_ind in range(len(class_heatmap_dict)):
            #     for dict_ind_cls_hm in class_heatmap_dict[str(dict_ind)]:
            #         dict_ind_cls_hm[0, 0, 0, :] = 255
            #         dict_ind_cls_hm[1, 0, 0, :] = 0

        mean_subtracted_heatmap = dict()
        # create heatmap for each patient
        # ind2: index for patient
        for ind2, (img, label, img_filename) in enumerate(zip(imgs_list, imgs_label, imgs_filename)):
            resized_img = imgip.interpolateImage(img) # (64, 97, 79, 1)

            # 1. descending sort    2. combine! (the number of classes - 1)*
            # sorted_class_heatmap = sorted(class_heatmap_dict.items(), key=operator.itemgetter(1), reverse = True)

            class_heatmap_to_sort = [(class_heatmap_dict[str(_c_ind)][ind2], _preds[ind2][_c_ind]) for _c_ind in range(self._num_classes)] # [(class_heatmap for img, pred), (), ()]
            sorted_class_heatmap = sorted(class_heatmap_to_sort, key=lambda x:x[1], reverse = True) #


            #img_cbined = np.zeros((resize_size[0], resize_size[1], resize_size[2], 3), dtype="float32") # resize_size ; (D, H, W)
            img_cbined = np.zeros(resize_size, dtype="float32")  # resize_size ; (D, H, W)
            for _c_ind in range(self._num_classes):
                if _c_ind == 0:
                    img_cbined += (self._num_classes-1) * sorted_class_heatmap[_c_ind][0]
                else:
                    img_cbined -= sorted_class_heatmap[_c_ind][0]

            # img2 += img_cbined
            # img_cbined /= np.max(np.abs(img_cbined))
            #img_cbined = img_cbined / np.max(np.abs(img_cbined))
            img_cbined = img_cbined / np.max(np.maximum(img_cbined, 0))
            img_cbined = np.clip(img_cbined, 0, 1)
            img_cbined *= 255
            img_cbined = np.uint8(img_cbined)

            # if not os.path.isdir(os.path.join(CAM_save_dir, "npy_img_cbined_" + str(label))):
            #     os.mkdir(os.path.join(CAM_save_dir, "npy_img_cbined_" + str(label)))
            # npy_save_path = os.path.join(CAM_save_dir, "npy_img_cbined_" + str(label))
            # save_filename = imgs_filename[ind2] + "_heatmap_.npy"
            # npy_mat_saver(img_cbined, os.path.join(npy_save_path, save_filename))
            if not os.path.isdir(os.path.join(CAM_save_dir, "npy_img_origin_" + str(label))):
                os.mkdir(os.path.join(CAM_save_dir, "npy_img_origin_" + str(label)))
            npy_save_path = os.path.join(CAM_save_dir, "npy_img_origin_" + str(label))
            save_filename = imgs_filename[ind2] + "_origin_.npy"
            npy_mat_saver(resized_img, os.path.join(npy_save_path, save_filename))

            img_cbined = thresholding_matrix(mat_to_mask=img_cbined, std_mat=None, c_ratio=0.4)
            #img_cbined[:, 0, 0] = 0

            model_pred = np.argmax(_preds[ind2])
            try:
                mean_subtracted_heatmap[str(model_pred)].append(img_cbined)
            except KeyError as ke:
                mean_subtracted_heatmap[str(model_pred)] = []
                mean_subtracted_heatmap[str(model_pred)].append(img_cbined)

            # try:
            #     mean_subtracted_heatmap[str(model_pred)].append(img_cbined)
            # except KeyError as ke:
            #     mean_subtracted_heatmap[str(model_pred)] = []
            #     mean_subtracted_heatmap[str(model_pred)].append(img_cbined)

            # Save Class Activation Map according to the each label
            if not os.path.isdir(os.path.join(CAM_save_dir, str(label))):
                os.mkdir(os.path.join(os.path.join(CAM_save_dir, str(label))))
            CAM_save_path = os.path.join(CAM_save_dir, str(label),
                                         img_filename + "_CAM_" + self._class_name[0] + self._class_name[
                                             1] + "_combined" + ".png")

            # proposed heatmap
            img1 = resized_img[:, :, :, 0]
            img1 = img1.astype(np.uint8)
            img1[:, 0, 0] = 0
            img1[:, 0, 1] = 255
            # plt.imshow(img1[32], cmap='gist_gray')
            # plt.show()
            #
            # ind = np.argmax([_img_cbined.max() for _img_cbined in img_cbined]) # (D, H, W)
            # plt.imshow(img_cbined[ind], cmap=plt.get_cmap('jet'), alpha=0.5)
            # plt.show()


            predictions_ = [str(round(item_ * 100, 2)) for item_ in _preds[ind2]]
            _title = "CAM: difference" + ", " + "GT:" + self._class_name[label] + ", " + \
                     " ".join([c_name + ":" + predictions_[ind] for ind, c_name in enumerate(self._class_name)])
            draw_img_on_grid([img1, img_cbined],
                             save_path=CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

            # index for CAM
            for _c_ind in range(self._num_classes):
                predictions_ = [str(round(item_*100, 2)) for item_ in _preds[ind2]]
                #print("debug predictions", predictions_) # ['0.0', '1.0', '0.0']

                _title = "CAM:"+self._class_name[_c_ind]+", "+"GT:"+self._class_name[label]+", "+\
                         " ".join([c_name+":"+predictions_[ind] for ind, c_name in enumerate(self._class_name)])
                #print("debug", _title)

                # Save Class Activation Map according to the each label
                if not os.path.isdir(os.path.join(CAM_save_dir, str(label))):
                    os.mkdir(os.path.join(os.path.join(CAM_save_dir, str(label))))
                CAM_save_path = os.path.join(CAM_save_dir, str(label), img_filename+"_CAM_"+self._class_name[_c_ind]+".png")

                img1 = resized_img[:, :, :, 0] # (D, H, W)
                img1 = img1.astype(np.uint8)
                img1[:, 0, 0] = 0
                img1[:, 0, 1] = 255
                img2 = class_heatmap_dict[str(_c_ind)][ind2]
                #print("[!] debug img2 min max", img2.min(), img2.max())
                #ind = np.argmax([_img_cbined.max() for _img_cbined in img2])  # (D, H, W)
                # draw_img_on_grid([img2],
                #                  save_path=None, is2D=False, _input_img_alpha=None, _overlap_img_alpha=None,
                #                  _title="test", _input_img_cmap=plt.get_cmap("jet"))

                if not os.path.isdir(os.path.join(CAM_save_dir, "npy_" + str(label))):
                    os.mkdir(os.path.join(CAM_save_dir, "npy_" + str(label)))
                npy_save_path = os.path.join(CAM_save_dir, "npy_" + str(label))
                save_filename = imgs_filename[ind2] + str(_c_ind)+"_heatmap_.npy"
                npy_mat_saver(img2, os.path.join(npy_save_path, save_filename))
                #img2 = [thresholding_matrix(mat_to_mask=_img2, std_mat=None, c_ratio=0.7) for _img2 in img2]
                if not os.path.isdir(os.path.join(CAM_save_dir, "npy_img_origin_" + str(label))):
                    os.mkdir(os.path.join(CAM_save_dir, "npy_img_origin_" + str(label)))
                npy_save_path = os.path.join(CAM_save_dir, "npy_img_origin_" + str(label))
                save_filename = imgs_filename[ind2] + "_origin_.npy"
                npy_mat_saver(img1, os.path.join(npy_save_path, save_filename))
                img2 = thresholding_matrix(mat_to_mask=img2, std_mat=None, c_ratio=0.4)
                #img2 = np.array(img2)
                #img2[0,:,:,:]=0
                #img2[:, 0, 0] = 0
                #print("[!] debug img2 min max", img2.min(), img2.max())
                # draw_img_on_grid([img2],
                #                  save_path=None, is2D=False, _input_img_alpha=None, _overlap_img_alpha=None,
                #                  _title="test", _input_img_cmap=plt.get_cmap("jet"))

                # original heatmap
                draw_img_on_grid([img1, img2],
                                 save_path=CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

                #img2 = np.zeros((64, 97, _c_ind), dtype="float32")
                # print("img debug", np.array(img).shape)

                pred_ind = _preds[ind2].argmax()
                if label == pred_ind :
                    # Save Class Activation Map only for a correct prediction
                    if not os.path.isdir(os.path.join(CAM_save_dir, "correct_" + str(label))):
                        os.mkdir(os.path.join(os.path.join(CAM_save_dir, "correct_" + str(label))))
                    CAM_save_path = os.path.join(CAM_save_dir, "correct_" + str(label),
                                                 img_filename + "_CAM_" + self._class_name[_c_ind] + ".png")
                    draw_img_on_grid([img1, img2],
                                 save_path=CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title
                    ,_input_img_cmap = "gist_gray", _overlap_img_cmap = plt.get_cmap("jet"))

                elif label != pred_ind :
                    # Save Class Activation Map only for a incorrect prediction
                    if not os.path.isdir(os.path.join(CAM_save_dir, "incorrect_" + str(label))):
                        os.mkdir(os.path.join(os.path.join(CAM_save_dir, "incorrect_" + str(label))))
                    CAM_save_path = os.path.join(CAM_save_dir, "incorrect_" + str(label),
                                                 img_filename + "_CAM_" + self._class_name[_c_ind] + ".png")
                    draw_img_on_grid([img1, img2],
                                     save_path=CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

        # Save Mean CAM for each of class label
        for _c_ind in range(self._num_classes):
            mean_cam_c = np.array(class_heatmap_dict[str(_c_ind)]).mean(axis=0)
            print("mean_cam_c", mean_cam_c.shape)
            _title = "rawdata + Mean CAM for each of class label" + str(_c_ind)
            rawdata_mean_CAM_save_path = os.path.join(CAM_save_dir, "rawdata+mean_CAM_"+self._class_name[_c_ind]+".png")

            img = imgip.interpolateImage(imgs_list[0])
            # print("img debug", np.array(img).shape)
            img1 = img[:, :, :, 0]
            img1 = img1.astype(np.uint8)
            img1[:, 0, 0] = 0
            img1[:, 0, 1] = 255
            draw_img_on_grid([img1, mean_cam_c],
                             save_path=rawdata_mean_CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

            _title = "Mean CAM for each of class label"+str(_c_ind)
            mean_CAM_save_path = os.path.join(CAM_save_dir, "mean_CAM_" + self._class_name[_c_ind] + ".png")
            draw_img_on_grid([mean_cam_c],
                             save_path=mean_CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

        # Save Mean subtracted CAM for each of prediction of model_name
        for _p_ind in range(self._num_classes):
            mean_cam_p = np.array(mean_subtracted_heatmap[str(_p_ind)]).mean(axis=0)
            print("mean_cam_p", mean_cam_p.shape)
            _title = "rawdata + Mean subtracted CAM for each of class label" + str(_p_ind)
            rawdata_mean_subtracted_CAM_save_path = os.path.join(CAM_save_dir,
                                                      "rawdata+mean_subtracted_CAM_" + self._class_name[_p_ind] + ".png")

            img = imgip.interpolateImage(imgs_list[0])
            # print("img debug", np.array(img).shape)
            img1 = img[:, :, :, 0]
            img1 = img1.astype(np.uint8)
            img1[:, 0, 0] = 0
            img1[:, 0, 1] = 255
            draw_img_on_grid([img1, mean_cam_p],
                             save_path=rawdata_mean_subtracted_CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4,
                             _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))

            _title = "Mean subtracted CAM for each of class label" + str(_p_ind)
            mean_subtracted_CAM_save_path = os.path.join(CAM_save_dir, "mean_subtracted_CAM_" + self._class_name[_p_ind] + ".png")
            draw_img_on_grid([mean_cam_p],
                             save_path=mean_subtracted_CAM_save_path, is2D=False, _input_img_alpha=None, _overlap_img_alpha=0.4, _title=_title,
                             _input_img_cmap="gist_gray", _overlap_img_cmap=plt.get_cmap("jet"))




        del class_heatmap_dict

        return


    def _create_cv_file(self, data, label, num_k = 4, save_path=None):
        if save_path is None :

            test_size = 1.0 / float(num_k)
            #sss_inner = StratifiedShuffleSplit(n_splits=num_k, test_size=test_size, random_state=1)  # random_state is generated using np.random
            sss_inner = StratifiedShuffleSplit(n_splits=num_k, test_size=test_size, random_state=1)
            kf = KFold(n_splits=num_k)

            cv_index_train = dict()
            cv_index_test = dict()
            # sss_inner.get_n_splits(data, label)
            #for ind, (train_index, test_index) in enumerate(sss_inner.split(data, label)):
            for ind, (train_index, test_index) in enumerate(kf.split(data, label)):
                print("train", len(train_index))
                print("test", len(test_index))
                cv_index_train["train_"+str(ind)] = train_index.tolist()
                cv_index_test["test_"+str(ind)] = test_index.tolist()

            pd_cv_index_train = pd.DataFrame(cv_index_train)
            pd_cv_index_train.to_excel(os.path.join(self._root_dir, self._result_save_dir, "cv_file_train.xlsx"))
            pd_cv_index_test = pd.DataFrame(cv_index_test)
            pd_cv_index_test.to_excel(os.path.join(self._root_dir, self._result_save_dir, "cv_file_test.xlsx"))
        else:
            pd_cv_index_train = pd.read_excel(os.path.join(self._root_dir, self._result_save_dir, "cv_file_train.xlsx"))
            pd_cv_index_test = pd.read_excel(os.path.join(self._root_dir, self._result_save_dir, "cv_file_test.xlsx"))
        return pd_cv_index_train, pd_cv_index_test

    def _plot_trend(self, trend_dict, save_path):

        # for item in trend_dict.items(): # key value pair i.g. "train_acc": 90.0
        #     plt.plot(item[1], label=item[0])
        # plt.legend('upper right')
        # plt.savefig(save_path)
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(trend_dict["train_loss_list"], 'y', label='train loss')
        loss_ax.plot(trend_dict["test_loss_list"], 'r', label='val loss')
        acc_ax.plot(trend_dict["train_acc_list"], 'b', label='train acc')
        acc_ax.plot(trend_dict["test_acc_list"], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.savefig(save_path)
        plt.close("all")
        return

    def _measure_acc(self, pred_ind_list, label_ind_list):

        return

    def get_output_layer(self, model, layer_name):
        # # get the symbolic outputs of each "key" layer (we gave them unique names).
        # layer_dict = dict([(layer.name, layer) for layer in model_name.layers])
        # # print("layer dict", layer_dict)
        # layer = layer_dict[layer_name]
        # return layer
        for layer in model.layers:
            if layer.name == layer_name:
                return layer

    def calculate_3D_CAM(self, X_data, _model, num_classes, featuremap_name, resize_size=None,
                         experiment_name_for_preprocess=None, preprocess_save_path=None):
        # preprocessing
        if experiment_name_for_preprocess:
            preprocessed_imgs_list = self._preprocess(X_data, isTrain=False,
                                                      experiment_name=experiment_name_for_preprocess)
        elif preprocess_save_path:
            preprocessed_imgs_list = self._preprocess(X_data, isTrain=False,
                                                      save_path=preprocess_save_path)
        else:
            preprocessed_imgs_list = X_data

        cams = [] # (N, C, D, H, W)
        for ind, img in enumerate(preprocessed_imgs_list):

            original_img = np.array([img])

            origin_heatmap_c = []
            for _c_ind in range(num_classes):  # iteration on each class

                _class_output = self._model.output[:, _c_ind]  # (N, num_classes)

                # Get the last layer's feature map
                final_conv_layer = self.get_output_layer(_model, featuremap_name)  # (1, D, H, W, C)

                # Get the last layer's input weights to the softmax.
                grads = K.gradients(_class_output, final_conv_layer.output)[0] # gradients function considers list of gradients

                pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))  # (C, )
                iterate = K.function([self._model.input], [pooled_grads, final_conv_layer.output[0]])
                pooled_grads_value, conv_layer_output_value = iterate([original_img])
                for i, w in enumerate(range(len(pooled_grads_value))):
                    conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]

                heatmap = np.mean(conv_layer_output_value, axis=-1)

                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)

                origin_heatmap_c.append(heatmap)

            origin_heatmap_c = np.array(origin_heatmap_c)
            print("origin_heatmap_c", origin_heatmap_c.shape)  # (3, 4, 4, 4)

            if resize_size is not None:

                resized_heatmap = []
                # norm each heatmap to [0, 1]
                for h_ind, hm in enumerate(origin_heatmap_c):
                    imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=resize_size)
                    hm = imgip.interpolateImage(hm)

                    resized_heatmap.append(hm)

                origin_heatmap_c = np.array(resized_heatmap)

            cams.append(origin_heatmap_c)

        return np.array(cams)
