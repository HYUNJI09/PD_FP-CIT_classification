import os
import numpy as np
from scipy import stats
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
from sklearn import linear_model

from DataIO import NIIDataIO
from ImageDataIO import *
from preprocess import *

class MarginalTest():
    def __init__(self, target_path=None, dtype="img"):
        self._samples = None
        self._labels = None
        return

    def updateData(self, datas, labels):
        self._samples = datas
        self._labels = labels
        return


    def create_p_matrix(self, x1, x2, type_test='anova'):
        """

        :param x1: (N, F) or (N, X, Y)
        :param x2: (N, F) or (N, X, Y)
        :param type_test:
        :return: statistics like (F,) or (X, Y)
        """
        p_matrix = None
        if type_test == 'anova':
            print("[!] create p-value matrix with ANOVA")
        elif type_test == 'ttest_ind':
            print("[!] create p-value matrix with ttest_ind")
            statstics = stats.ttest_ind(x1, x2, axis=0)
        elif type_test == 'paired_t_test':
            print("[!] create p-value matrix with paired_t_test")

        return statstics


    def fdr_masking(self, p_vals, alpha=0.05, method='fdr_bh'):
        """
        :param p_vals:
        :param alpha:
        :param method:
        :return: (reject, pvals_corrected)
        """

        p_vals_flat = p_vals.flatten()
        res = multitest.multipletests(pvals=p_vals_flat, alpha=alpha, method=method, is_sorted=False, returnsorted=False)

        return res[0].reshape(p_vals.shape), res[1].reshape(p_vals.shape)

    def export_excel(self, save_path):

        return

class Lasso():
    def __init__(self):
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._model = None

    def train_lasso_eval(self, X_train, y_train, X_test=None, y_test=None, C=0.05):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        #lasso_reg = linear_model.Lasso(alpha=C, tol=1e-4)
        lasso_reg = linear_model.LogisticRegression(penalty='l1', C=C, solver='liblinear')
        lasso_reg.fit(X_train, y_train)
        self._model = lasso_reg

        print("train score", self._model.score(X_train, y_train))
        if X_test is not None:
            print("test score", self._model.score(X_test, y_test))
        return

    def get_coef(self):
        return self._model.coef_

def atlas_based_eval_pval_mask(pval_mask, bin_target_region_filepath_list, target_value_list= None):
    num_feature_sel_list = []
    filename_list = []
    for ind, target_region_filepath in enumerate(bin_target_region_filepath_list):
        target_region_filepath
        extension = "nii"
        dataDim = "3D"  # 3D
        instanceIsOneFile = True

        modeling = "3D"
        view = "axial"

        idio = ImageDataIO(extension, dataDim=dataDim, instanceIsOneFile=instanceIsOneFile, modeling="3D",
                           view=view)
        _data = idio.read_file(target_region_filepath)
        final_target_region = np.zeros(_data.shape)
        target_val = target_value_list[ind]
        final_target_region[_data==target_val]=1

        selected_f_mask_on_target_region = pval_mask*final_target_region
        num_feature_sel_list.append(len(selected_f_mask_on_target_region[selected_f_mask_on_target_region==1]))
        filename_list.append(os.path.basename(target_region_filepath))
    return num_feature_sel_list, filename_list

if __name__ == "__main__":
    mode = "atlas_based_eval_pval_mask" # "Voxel-wise marginal test"
    if mode== "atlas_based_eval_pval_mask":
        p_val_mask = np.ones([68, 95, 79])

        bin_target_region_filepath_list = [r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Frontal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Parietal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Posterior_Cingulate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Temporal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Cerebellum_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                                           r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Hammers_WholeBrain_2mm_79_95_68.nii']

        target_value_list = [1, 1, 1, 1, 1, 1, 0]
        test_result = atlas_based_eval_pval_mask(p_val_mask, bin_target_region_filepath_list, target_value_list=target_value_list)
        print(test_result)
    elif mode == "Voxel-wise marginal test":
        print("[!] Voxel-wise marginal test")

        # Loading data
        data_dir = r"C:\Users\hkang\PycharmProjects\datas\ADNI\arranged_imgs_sylee_nii_hk\2_CN_SN_raw_nii"
        extension = "nii"
        dataDim = "3D"  # 3D
        modeling = "3D"
        view = "axial"

        input_shape = (64, 64, 64)
        channel_size = 1

        nii_io = NIIDataIO()
        nii_io.load(data_dir, extension=extension, data_dir_child="labeled",
                    dataDim=dataDim, instanceIsOneFile=True, channel_size=channel_size, view=view)

        datas = nii_io._3D_data
        labels = nii_io._label
        print(np.array(nii_io._3D_data).shape)  # (213, 68, 95, 79, 1)
        print(np.array(nii_io._3D_data_filename).shape)  # (256,)
        print(np.array(nii_io._label).shape)  # (256,)
        print(np.array(nii_io._label_name).shape)  # (256,)
        print(np.array(nii_io._class_name))  # ['0_bapl1' '1_bapl23']

        # preprocessing
        scaled_datas = apply_min_max_normalization(datas)
        scaled_datas = scaled_datas*255
        #_, scaled_datas = StandardScalingData(datas, save_path=None, keep_dim=True, train=True, scaler=None)

        # split data by group label
        bapl1_nii = scaled_datas[labels == 0]
        bapl23_nii = scaled_datas[labels == 1]
        print(bapl1_nii.shape)  # (157, 68, 95, 79)
        print(bapl23_nii.shape)  # (99, 68, 95, 79)

        feature_selector = 'Lasso' # 'Lasso'
        if feature_selector == 'Marginal_test':
            #bapl1_nii = (bapl1_nii-bapl1_nii.min(axis=0))/bapl1_nii.max(axis=0) * 255
            #bapl23_nii = (bapl23_nii - bapl23_nii.min(axis=0)) / bapl23_nii.max(axis=0) * 255

            #nii_io.show_one_img_v3(bapl1_nii[0], is2D=False, cmap=plt.get_cmap('hot'))
            # nii_io.show_one_img_v3(bapl1_nii[0], is2D=False, cmap=plt.get_cmap('gray'))

            # calculating p-value matrix
            m_testor = MarginalTest()
            statistics = m_testor.create_p_matrix(bapl1_nii, bapl23_nii, type_test='ttest_ind')
            # nii_io.show_one_img_v3(statistics[0], is2D=False, cmap=plt.get_cmap('gray'))
            # nii_io.show_one_img_v3(statistics[1], is2D=False, cmap=plt.get_cmap('gray'))


            statistics = m_testor.fdr_masking(statistics[1], alpha=0.05, method='fdr_bh')  # 'bonferroni'
            pval_map = statistics[1]
            nii_io.show_one_img_v3(pval_map, is2D=False, cmap=plt.get_cmap('gray'))
        elif feature_selector == 'Lasso':
            lasso_ref = Lasso()

            data_shape = datas.shape
            # flatten data
            flatten_datas = datas.reshape([data_shape[0], data_shape[1]*data_shape[2]*data_shape[3]])
            print("flatten_datas.shape", flatten_datas.shape)
            lasso_ref.train_lasso_eval(X_train=flatten_datas, y_train=labels, X_test=None, y_test=None, C=50.0)

            betas = lasso_ref.get_coef()
            print("betas.shape", betas.shape)
            reshaped_betas  = betas.reshape(data_shape[1], data_shape[2], data_shape[3])
            print("test np.unique(reshaped_betas)", np.unique(reshaped_betas).shape, reshaped_betas.min(), reshaped_betas.max())

            mask_map = np.zeros(reshaped_betas.shape)
            mask_map[reshaped_betas == 0] = 0
            mask_map[reshaped_betas != 0] = 1

            print("test np.unique(mask_map)", np.unique(mask_map).shape)
            nii_io.show_one_img_v3(mask_map, is2D=False, cmap=plt.get_cmap('gray'))

            # reshaped_betas = apply_min_max_normalization([reshaped_betas])
            #nii_io.show_one_img_v3(reshaped_betas, is2D=False, cmap=plt.get_cmap('gray'))
        # pval_map[pval_map >= 0.05] = 2 ** 10
        # pval_map[pval_map < 0.05] = 1
        # pval_map[pval_map == 2 ** 10] = 0


        # print(statistics[0].shape)
        # print(statistics[1].shape)

