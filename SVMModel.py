# feature extraction : densenet
# classifier : SVM
import os
import numpy as np
import pandas as pd
import tqdm

from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler

# SVM
from sklearn import svm

# evaluate for SVM model_name
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# load modules for you to apply bayesian optimization algorithm
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from skopt import BayesSearchCV

from ImageDataIO import ImageDataIO
from preprocess import StandardScalingData
from data_load_utils import *


def make_onehot_vector(ys, num_classes):
    """
    :param ys: (N, C)
    :return:
    """
    class_ind = list(range(num_classes))
    onehot_vector_list = []
    for y_elem in ys:
        tmp_onehot = np.zeros(num_classes)
        y_c_ind = class_ind.index(y_elem)
        tmp_onehot[y_c_ind] = 1

        onehot_vector_list.append(tmp_onehot)
    return np.array(onehot_vector_list)

class SVM():
    def __init__(self, kernel='linear', C=None, gamma=None, seed = None, probability=None):
        # hyper parameters
        self._kernel = kernel
        self._C = C
        self._gamma = gamma
        self._seed = seed
        self._probability = probability

        self._datas = None
        self._labels = None
        self._data_filenames = None
        self._label_name = None
        self._class_name = None

        self._model = None
        self.clear_model()
        return

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
            self._datas = datas # It is not decided whether datas will be used for training or testing

        if labels is not None:
            self._labels = labels
            self._num_classes = len(np.unique(labels))

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
        #for ind in range(num_cv):
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

    def train_eval(self, X_train, y_train, X_test, y_test, X_train_filename=None, X_test_filename=None, output_save_dir=None):
        self.clear_model()
        tmp_datas = self._datas
        tmp_labels = self._labels
        tmp_data_filenames = self._data_filenames
        tmp_label_name = self._label_name
        tmp_class_name = self._class_name


        self.LoadData(datas=X_train, labels=y_train, data_filenames=X_train_filename, label_name=self._label_name, class_name=self._class_name)
        train_accuracy, train_f1_score = self.train()

        self.LoadData(datas=X_test, labels=y_test, data_filenames=X_test_filename, label_name=self._label_name,
                      class_name=self._class_name)

        val_accuracy, val_f1_score = self.test(output_save_dir)

        self.LoadData(datas=tmp_datas, labels=tmp_labels, data_filenames=tmp_data_filenames, label_name=tmp_label_name,
                      class_name=tmp_class_name)

        return train_accuracy, train_f1_score, val_accuracy, val_f1_score


    def train(self):
        """
        :return: a model trained by loaded data / and the performance log
        """
        print("[!] SVM is prepared for training")
        print("kernel", self._kernel)
        print("C", self._C)
        print("gamma", self._gamma)

        self._model = svm.SVC(C=self._C, class_weight='balanced', gamma=self._gamma, decision_function_shape='ovr',
                        kernel=self._kernel, random_state=self._seed, probability=True)

        print("\n[!] Training SVM model_name\n")
        self._model.fit(self._datas, self._labels)  # non one hot style

        train_predict = self._model.predict(self._datas)
        # train_predict_proba = self._model.predict_proba(self._datas)
        train_accuracy = accuracy_score(train_predict, self._labels)
        train_f1_score = f1_score(train_predict, self._labels)
        print("Training Accuracy: {0:.6f}".format(train_accuracy))
        print("Training F1_score: {0:.6f}".format(train_f1_score))
        return train_accuracy, train_f1_score


    def test(self, output_save_dir):
        """

        :param output_save_dir:
        :return:
        """

        val_predict = self._model.predict(self._datas)  # return val shape : (n_samples,)
        val_predict_proba = self._model.predict_proba(self._datas)  # return val shape : (n_samples,)
        val_accuracy = accuracy_score(val_predict, self._labels)
        val_f1_score = f1_score(val_predict, self._labels)
        print("Validation Accuracy: {0:.6f}".format(val_accuracy))
        print("Validation F1_score: {0:.6f}".format(val_f1_score))

        # Evaluation
        print('Test phase')
        if output_save_dir is not None and not os.path.isdir(output_save_dir):
            os.mkdir(output_save_dir)
        if output_save_dir:
            save_filename = output_save_dir + "\\" + "final_models_ROC_data.xlsx"

        if output_save_dir:
            onehot_labels = make_onehot_vector(self._labels, len(np.unique(self._labels)))
            save_pred_label(onehot_labels, val_predict_proba, onehot_label=True, save_filepath=save_filename,
                            filename_list=self._data_filenames)  # y_test_ : (N, Num_classes)

        conf_m = confusion_matrix(self._labels, val_predict)
        print("CV #", "confusion matrix")
        print(conf_m)

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


        return val_accuracy, val_f1_score



if __name__ == "__main__":
    print("[*] Experiment Start")
    for ind in range(36):
        comp_index = str(ind)
        param_save_dir = "C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_comp_results\\SVM\\hyparam_fix\\augmented"+\
        "\\"+comp_index
        if not os.path.isdir(param_save_dir):
            os.mkdir(param_save_dir)

        d_svm = DenseSVM()

        # 0:bapl1 class, 1:bapl3 class
        test_dir_list = [["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\original\\bapl1"+"\\"+comp_index],
                          ["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\original\\bapl3"+"\\"+comp_index]]
        train_dir_list = [["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\training\\bapl1"+"\\"+comp_index,
                          "C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\generated\\bapl1"+"\\"+comp_index],
                         ["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\training\\bapl3"+"\\"+comp_index,
                          "C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\generated\\bapl3" + "\\" + comp_index]]

        # train_dir_list = [["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\training\\bapl1"+"\\"+comp_index],
        #                  ["C:\\Users\\NM\\PycharmProjects\\GAN_evaluation\\dataset_for_measure\\grayscale\\slice_sample\\training\\bapl3"+"\\"+comp_index]]
        # Load data
        d_svm.LoadData(train_dir_list, test_dir_list) # case which holdout is not necessary, holdout is already done

        # Experiment
        num_K = 4
        test_size = 1.0/float(num_K)
        acc = d_svm.model_selection_evaluate_experiment(num_K, test_size, output_save_dir=param_save_dir)
        del d_svm
