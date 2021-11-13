# feature extraction : densenet
# classifier : SVM
import os
import numpy as np
import pandas as pd
import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
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
from GAN_eval_metrics import ConvNetFeatureSaver_Keras
from preprocess import StandardScalingData
from data_load_utils import *

class DenseSVM():
    def __init__(self):
        return

    # without holdout
    def LoadData(self, train_dir_list, test_dir_list):
        """
        train_dir_list and test_dir_list consist of list of dir for each class
        each of parameter is [Num_Classes, Num_dirs]
        the dirs on the Num_dirs are the root directories including images seperated
        :param train_dir_list: 
        :param test_dir_list: 
        :return: 
        """
        self._train_data = []
        self._train_label = []
        self._train_filename = []
        self._test_data = []
        self._test_label = []
        self._test_filename = []

        # read_train
        idio = ImageDataIO(extention="png", is2D=True, view="axial")
        for class_label, _dir_list in enumerate(train_dir_list):
            for _dir in _dir_list:
                img_list, filename_list = idio.read_files_from_dir(_dir)
                img_list = idio._resizing_channel(img_list, (224, 224), channel_size=3)
                img_list = idio.convert_PIL_to_numpy(img_list)
                print("img_list shape", img_list.shape) # (2880, 224, 224, 3),    (2484, 224, 224, 3)
                self._train_data.append(img_list)
                self._train_label.append([class_label]*len(img_list))
                self._train_filename.append(filename_list)
        self._train_data = np.concatenate(self._train_data)
        self._train_label = np.concatenate(self._train_label)
        self._train_filename = np.concatenate(self._train_filename)
        # read_test
        idio = ImageDataIO(extention="png", is2D=True, view="axial")
        for class_label, _dir_list in enumerate(test_dir_list):
            for _dir in _dir_list:
                img_list, filename_list = idio.read_files_from_dir(_dir)
                img_list = idio._resizing_channel(img_list, (224, 224), channel_size=3)
                img_list = idio.convert_PIL_to_numpy(img_list)
                self._test_data.append(img_list)
                self._test_label.append([class_label] * len(img_list))
                self._test_filename.append(filename_list)
        self._test_data = np.concatenate(self._test_data)
        self._test_label = np.concatenate(self._test_label)
        self._test_filename = np.concatenate(self._test_filename)
        print("_train_data shape", np.array(self._train_data).shape)  # (5364, 224, 224, 3)
        print("_test_data shape", np.array(self._test_data).shape) # (5364, 224, 224, 3)

        scaler = StandardScaler()
        scaler, self._train_data = StandardScalingData(self._train_data, scaler, keep_dim=True, train=True, save_path=None)
        _, self._test_data = StandardScalingData(self._test_data, scaler, keep_dim=True, train=False, save_path=None)

        convfs = ConvNetFeatureSaver_Keras(model="densenet121")
        self._train_data = convfs.feature_extractor_from_npMat(self._train_data)
        self._test_data = convfs.feature_extractor_from_npMat(self._test_data)
        print("_train_data shape", self._train_data.shape)  #
        print("_test_data shape", self._test_data.shape)

        return

    def train(self):
        return

    # model_selection : with bayesian optimizer
    def model_selection_evaluate_experiment(self, num_K, test_size, output_save_dir=None):
        # best_acc_candidate = []
        # best_param_candidate = []
        # sss = StratifiedShuffleSplit(n_splits=num_K, test_size=test_size, random_state=None) # random_state is generated using np.random
        # sss.get_n_splits(self._train_data, self._train_label)
        # for ind, (train_index, test_index) in enumerate(sss.split(self._train_data, self._train_label)):
        #     self.X_train = self._train_data[train_index]
        #     self.y_train = self._train_label[train_index]
        #     # X_train_filename = X_data_filename[train_index]
        #
        #     self.X_test = self._train_data[test_index]
        #     self.y_test = self._train_label[test_index]
        #     # X_test_filename = X_data_filename[test_index]
        #
        #     print("X_train_path", self.X_train.shape)  # 269
        #     print("y_train", self.y_train.shape)  # 269
        #     print("X_test_path", self.X_test.shape)  # 90
        #     print("y_test", self.y_test.shape)  # 90

            # training data
            # num_classes = len(class_list)
            # print("class_list", class_list)  # ['gr1', 'gr2', 'gr3']

            # use of scikit-optimizer for Bayesian optimizer using GP
        #     dim_kernel = Categorical(categories=['rbf', 'linear', 'poly'], name='kernel')
        #     dim_C = Real(low=1, high=1e2, prior='log-uniform', name='C')
        #     dim_gamma = Real(low=1e-4, high=0.1, prior='log-uniform', name='gamma')  # gamma higher bound into 0.1
        #     # dim_num_PCA_feature = Integer(low=100, high=5000,
        #     #                               name='num_PCA_feature')  # ValueError: n_components=7590 must be between 0 and n_features=7505 with svd_solver='full' /?
        #     # PCA feature lower bound into 100
        #     dimensions = [dim_kernel, dim_C, dim_gamma]
        #     default_parameters = ['linear', 1, 0.1]
        #
        #     # set a path where you save a trained model_name
        #     # path_best_model = './results/300aug_vgg16_best.h5'
        #     self.best_accuracy = 0.0
        #     # best_loss = 0.0
        #     self.fitness_call_count = 0
        #
        #     log_dim_kernel = []
        #     log_dim_c = []
        #     log_dim_gamma = []
        #     # log_dim_num_PCA_feature = []
        #
        #     log_acc = []
        #     # log_explained_variance_ratio = []
        #
        #     fitness_call_count = 0
        #
        #     @use_named_args(dimensions=dimensions)
        #     def fitness(kernel, C, gamma):
        #         print('kernel : ', kernel)
        #         print('C : ', C)
        #         print('gamma : ', gamma)
        #         #print('num_PCA_feature : ', num_PCA_feature)
        #
        #         X_train = np.array(self.X_train)
        #         X_test = np.array(self.X_test)
        #         y_train = np.array(self.y_train)
        #         y_test = np.array(self.y_test)
        #
        #         print("X_train_inner_feature", self.X_train.shape)
        #         print("X_test_inner_feature", self.X_test.shape)
        #         print("y_train", self.y_train.shape)
        #         print("y_test", self.y_test.shape)
        #
        #         print("\n[!] Create SVM model_name with proposed hyper-parameter on the phase, model_name selection \n")
        #         model_name = svm.SVC(C=C, class_weight='balanced', gamma=gamma, decision_function_shape='ovr', kernel=kernel,
        #                         random_state=None)
        #         print("\n[!] Training SVM model_name\n")
        #         model_name.fit(self.X_train, self.y_train)  # non one hot style
        #         train_predict = model_name.predict(self.X_train)
        #         train_accuracy = accuracy_score(train_predict, self.y_train) * 100
        #
        #         val_predict = model_name.predict(self.X_test)
        #         accuracy = accuracy_score(val_predict, self.y_test) * 100
        #         print("Training Accuracy: {0:.6f}".format(train_accuracy))
        #         print("Validation Accuracy: {0:.6f}".format(accuracy))
        #
        #         if accuracy > self.best_accuracy:
        #             # new_model.save(path_best_model)
        #             self.best_accuracy = accuracy
        #
        #         log_dim_kernel.append(kernel)
        #         log_dim_c.append(C)
        #         log_dim_gamma.append(gamma)
        #         #log_dim_num_PCA_feature.append(num_PCA_feature)
        #
        #         log_acc.append(accuracy)
        #
        #         self.fitness_call_count += 1
        #         del model_name
        #         return -accuracy
        #         # end of fitness()
        #
        #     search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=20,
        #                                 x0=default_parameters)
        #     # plot_convergence(search_result)
        #
        #     # print("search_result.x : ", search_result.x)
        #     # print("search_result.fun : ", search_result.fun)
        #     best_acc_candidate.append(search_result.fun)
        #     best_param_candidate.append(search_result.x)
        #
        #     data = dict()
        #
        #     data['log_dim_kernel'] = log_dim_kernel
        #     data['log_dim_c'] = log_dim_c
        #     data['log_dim_gamma'] = log_dim_gamma
        #     #data['log_dim_num_PCA_feature'] = log_dim_num_PCA_feature
        #     # data['log_explained_variance_ratio'] = log_explained_variance_ratio
        #     data['log_acc'] = log_acc
        #
        #     df_data = pd.DataFrame(data)
        #     # log_save_path_name = os.path.join("./results", "outer_loop", "CV" + str(ind + 1), "inner_loop",
        #     #                                   "CV" + str(ind2 + 1), "no_aug_pca_svm_best_Bayesian_Op_log.csv")
        #     if output_save_dir:
        #         df_data.to_excel(output_save_dir+"\\"+"bayesian_op_log.xlsx")
        #
        # # # log best param & acc
        # # with open("./results/outer_loop/CV" + str(
        # #                 ind + 1) + "/no_aug_pca_svm_best_Bayesian_Op_log_results_of_inner_loop.txt", "w") as f:
        # #     f.write("acc\tparam\n")
        # #     for acc_ind in range(len(best_acc_candidate)):
        # #         f.write(str(best_acc_candidate[acc_ind]) + "\t" + str(best_param_candidate[acc_ind]) + "\n")
        #
        # best_acc_ind = np.argmin(best_acc_candidate)
        # best_param = best_param_candidate[best_acc_ind]
        #
        # best_kernel = best_param[0]
        # best_C = best_param[1]
        # best_gamma = best_param[2]
        best_kernel = "rbf"
        best_C = 100
        best_gamma = 0.0001
        # best_kernel = "linear"
        # best_C = 1
        # best_gamma = 0.1
        #best_num_PCA_feature = best_param[3]
        print("best_kernel", best_kernel)
        print("best_C", best_C)
        print("best_gamma", best_gamma)
        # with open(output_save_dir + "\\" + "best_model_parameter.txt", "w") as f:
        #     f.write("Accuracy : " + str(best_acc_candidate[best_acc_ind]) + '\n')
        #     for row in best_param:
        #         f.write("%s\n" % row)

        # print("[debug!!!!!!] y_train_inner", y_train_inner.shape)
        print("\n[!] Create SVM model_name with best hyper-parameter on outerloop\n")
        model = svm.SVC(C=best_C, class_weight='balanced', gamma=best_gamma, decision_function_shape='ovr',
                        kernel=best_kernel, random_state=None, probability=True)
        print("\n[!] Training SVM model_name\n")
        model.fit(self._train_data, self._train_label)  # non one hot style
        train_predict = model.predict(self._train_data)
        train_predict_proba = model.predict_proba(self._train_data)
        train_accuracy = accuracy_score(train_predict, self._train_label)

        val_predict = model.predict(self._test_data)  # return val shape : (n_samples,)
        val_predict_proba = model.predict_proba(self._test_data)  # return val shape : (n_samples,)
        accuracy = accuracy_score(val_predict, self._test_label) * 100
        print("Training Accuracy: {0:.6f}".format(train_accuracy))
        print("Validation Accuracy: {0:.6f}".format(accuracy))

        # Evaluation
        print('Test phase')
        if output_save_dir:
            save_filename = output_save_dir + "\\" + "final_models_ROC_data.xlsx"

        save_pred_label(self._test_label, val_predict_proba, onehot_label=False, save_filepath=save_filename,
                        filename_list=self._test_filename)  # y_test_ : (N, Num_classes)

        conf_m = confusion_matrix(self._test_label, val_predict)
        print("CV #", "confusion matrix")
        print(conf_m)

        # logging the conf_m
        with open(output_save_dir + "\\" + "best_model_performance_report_conf_m.txt", "w") as f:
            f.write("Accuracy : " + str(accuracy) + '\n')
            for row in conf_m:
                f.write("%s\n" % row)

        target_names = ['BAPL1', 'BAPL3']

        result = classification_report(self._test_label, val_predict, target_names=target_names)
        print("Test Phase result")
        print(result)
        with open(output_save_dir + "\\" + "best_model_performance_report_conf_m.txt", "a") as f:
            f.write("%s\n" % result)

        del model
        return accuracy

    def test(self):

        return


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

