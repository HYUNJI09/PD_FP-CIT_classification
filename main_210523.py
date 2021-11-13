## Data 분류 모델 만들기


# 0.라이브러리 호출 및 관련 함수 준비
import sys
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split    # train과 test 나눔 라이브러리
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc    # 모델 성능 확인 라이브러리
from Conv3DModel import *   # 3D CNN 구성 라이브러리
from DataIO import NIIDataIO    # nii data read 라이브러리
from ImageDataIO import *    # 이미지 데이터 읽기 라이브러리
from preprocess import *    # 데이터 전처리 라이브러리
from FeatureSelector import *   # 이미지 데이터 특징 추출 라이브러리

from SVMModel import *  #서포트 벡터 머신(SVM) 분류기를 사용하여 관측값 분류 라이브러리


def get_dataset_from_nii_files():   # nii 파일 dataset으로 만드는 함수
    return


def sort_elem_list_by_strID(target_list_to_sort, candi_list_to_sort, candi_list_to_refer):
    # 파일마다 가지고 있는 이름을 가져와서 리스트로 만들기
    if type(target_list_to_sort).__module__ == np.__name__:
        target_list_to_sort = target_list_to_sort.tolist()
    if type(candi_list_to_sort).__module__ == np.__name__:
        candi_list_to_sort = candi_list_to_sort.tolist()
    if type(candi_list_to_refer).__module__ == np.__name__:
        candi_list_to_refer = candi_list_to_refer.tolist()

    sorted_target_list = [0]*len(target_list_to_sort)
    for ind, candi_elem in enumerate(candi_list_to_sort):
        target_ind = candi_list_to_refer.index(candi_elem)
        sorted_target_list[target_ind] = target_list_to_sort[ind]

    return np.array(sorted_target_list)


def get_performances(target, output, pos_label=1):  # 모델 성능 평가 함수
    # Evaluation 1 : Accuracy
    _acc = accuracy_score(target, output)

    # Evaluation 2 : AUROC
    fpr, tpr, _ = roc_curve(target, output, pos_label=pos_label)
    _auroc = auc(fpr, tpr)

    # Evaluation 3 : F1-score
    _f1_score = f1_score(target, output)

    # Evaluation 4 : geometric mean
    conf_mat = confusion_matrix(target, output)
    specificity = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    sensitivity = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    _g_mean = stats.gmean([specificity, sensitivity])

    return _acc, _auroc, _f1_score, _g_mean



# 1.데이터 불러오기.
if __name__ == "__main__":

    task_mode = "3D_CNN_model_test"
    # "3D_CNN_model_test_with_multiple_comparison" "3D_CNN_model_test_with_ft_sel"  # "3D_CNN_model_sel" / "3D_CNN_model_test" / "SVM" / "DNN_experiment" / "3D_CNN_model_test_with_ft_sel"
    # 사용할 데이터 모델 가져오기 - 3D CNN 모델에서도 다중 비교
    sys.setrecursionlimit(1100) # 이 제한은 무한 재귀로 인해 C 스택이 오버플로되고 Python이 충돌하는 것을 방지

    # Data loader
    # Loading data
    data_dir = r"D:\test"  # 데이터 불러오기: 폴더 별로 다르게 분류된 데이터로 인식
    extension = "nii"   # 폴더 내 인식할 데이터 파일형식: dicom or nii
    dataDim = "3D"  # 인식할 데이터의 차원: 2D or 3D
    view = "axial"  # brain image 영상의 방향 axial or coronal ...

    channel_size = 1    # 데이터 채널 설정

    nii_io = NIIDataIO()    # nii data를 불러오기 위한 빈 list 만들기
    nii_io.load_v2(data_dir, extension=extension, data_dir_child="labeled",
                dataDim=dataDim, instanceIsOneFile=True, channel_size=channel_size, view=view)
    # load_v2는 DataIO의 내장함수.
    # 이 코드는 최소-최대 정규화를 포함하고 입력 이미지를 (400, 400)에서 (200, 200)으로 줄임
    # data_dir_child: labeled" 또는 "sample",
    #   인수 "labeled"는 data_dir이 샘플 파일의 레이블이 지정된 디렉토리를 포함하는 상위 디렉토리임을 의미
    #   "sample"은 data_dir이 샘플 데이터(파일) 세트로 직접 구성됨을 의미

    datas = nii_io._3D_data # (N, D, W, H), (397, 110, 200, 200)
    labels = nii_io._label # (N, ) [0, 1, ...], (397, ) 라벨로 분류된 하위 폴더의
    data_filenames = nii_io._3D_data_filename # (N, ), (397, )
    label_name = nii_io._label_name # (N, ) ['0_NC', '1_PD', ..], (397, )
    class_name = nii_io._class_name # (Num_classes, ), (2, ), ['0_NC' '1_PD']
    print(np.array(nii_io._3D_data).shape)  # (213, 68, 95, 79, 1)
    print(np.array(nii_io._3D_data_filename).shape)  # (256,)
    print(np.array(nii_io._label).shape)  # (256,)
    print(np.array(nii_io._label_name).shape)  # (256,)
    print(np.array(nii_io._class_name))  # ['0_bapl1' '1_bapl23']

    # # 2. preprocessing
    # scaled_datas = apply_min_max_normalization(datas)
    # scaled_datas = scaled_datas * 255


    if task_mode == "3D_CNN_model_test_with_multiple_comparison":
        print("Conv3DModel 3D_CNN_model_test_with_multiple_comparison")

        EXPERIMENTAL_NAME = "./3D_CNN_apl1vs23_210708_raw_data_multiple_comparison"  # 'preprecess', trained model, evaluation log
        # preprocess_save_path = os.path.join(EXPERIMENTAL_NAME, 'model', EXPERIMENTAL_NAME[2:], "standardscaler.pkl")
        preprocess_save_path = None

        extension = "nii"
        dataDim = "3D"  # 3D
        modeling = "3D"
        view = "axial"

        lr_rate = 0.00005
        num_classes = 2
        # num_epochs = 1000
        num_epochs = 100 #200
        input_shape = (64, 64, 64)  # (D, H, W)
        channel_size = 1
        batch_size = 64


        # preprocessing before modeling : resize, channel
        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=input_shape)
        resized_scaled_datas = [imgip.interpolateImage(img) for img in datas]
        if resized_scaled_datas[0].shape[-1] != 1 or resized_scaled_datas[0].shape[-1] != 3:
            resized_scaled_datas = np.array(resized_scaled_datas)[:, :, :, :, None]

        # Split data into training and testing
        X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test = train_test_split(resized_scaled_datas,
                                                                                                       labels,
                                                                                                       data_filenames,
                                                                                                       train_size=0.70,
                                                                                                       random_state=1)
        print("X_train.shape", X_train.shape)  # X_train.shape (277, 64, 64, 64, 1)
        print("X_test.shape", X_test.shape)  # X_test.shape (120, 64, 64, 64, 1)
        print("data_filenames_train.shape", data_filenames_train.shape)  # X_test.shape (77, 68, 95, 79)

        # # split data by group label
        # bapl1_nii = scaled_datas[labels == 0]
        # bapl23_nii = scaled_datas[labels == 1]
        # print(bapl1_nii.shape)  # (157, 68, 95, 79)
        # print(bapl23_nii.shape)  # (99, 68, 95, 79)
        #
        # conv3dmodel.load3DImage(data_dir, extension=extension, data_dir_child="labeled",
        #                         dataDim=dataDim, instanceIsOneFile=True, view=view)

        # multiple comparison with bootstrapping
        perform_dist_savepath = r"./3D_CNN_apl1vs23_210707_5_raw_data_multiple_comparison/perform_dist.xlsx"
        num_bootstrapping = 100
        accuracy_list = []
        auroc_list = []
        f1_score_list = []
        gmean_list = []
        for ind in range(num_bootstrapping):

            train_ind_list = np.arange(1, len(X_train))
            subsampled_indices = np.random.choice(train_ind_list, replace=True, size=len(X_train))
            subsampled_X_train = X_train[subsampled_indices]
            subsampled_y_train = y_train[subsampled_indices]
            subsampled_data_filenames_train =data_filenames_train[subsampled_indices]
            conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, input_shape=input_shape,
                                      channel_size=channel_size, num_epochs=num_epochs,
                                      root_dir=EXPERIMENTAL_NAME, model_save_dir="models"
                                      , result_save_dir="results", batch_size=batch_size, cv_file=None,
                                      numAdditionalFeatures=None, model_reload_path=None)
            conv3dmodel.updateInput(datas=subsampled_X_train, labels=subsampled_y_train, data_filenames=subsampled_data_filenames_train, label_name=label_name,
                                    class_name=class_name)
            # conv3dmodel.train_eval_CV(num_cv=4)
            conv3dmodel.train_eval(subsampled_X_train, subsampled_y_train, X_test, y_test, EXPERIMENTAL_NAME, save_model_name=None)

            # s, _ = StandardScalingData(X_train, save_path=None, train=True, keep_dim=True)
            # _, scaled_X_test = StandardScalingData(X_test, save_path=None, train=False, keep_dim=True, scaler=s)
            # conv3dmodel.test(scaled_X_test, y_test, EXPERIMENTAL_NAME, data_filenames_test, preprocess_save_path=preprocess_save_path)
            _preds = conv3dmodel.test(X_test, y_test, EXPERIMENTAL_NAME, data_filenames_test)
            preds_hat = np.argmax(_preds, axis=1)
            acc, auroc, f1, g_mean = get_performances(y_test, preds_hat)
            accuracy_list.append(acc)
            auroc_list.append(auroc)
            f1_score_list.append(f1)
            gmean_list.append(g_mean)

            del conv3dmodel

        print("Bootstrapping ACC", np.mean(accuracy_list))
        print("Bootstrapping AUROC", np.mean(auroc_list))
        print("Bootstrapping F1-score", np.mean(f1_score_list))
        print("Bootstrapping gmean", np.mean(gmean_list))

        perform_dist_df = pd.DataFrame({"acc":accuracy_list, "auroc":auroc_list, "f1":f1_score_list, "gmean":gmean_list})
        perform_dist_df.to_excel(perform_dist_savepath)


    elif task_mode == "3D_CNN_model_test":
        print("Conv3DModel 3D_CNN_model_test")

        feature_selector = None  # 'Marginal_test', 'Lasso'

        EXPERIMENTAL_NAME = "./3D_CNN_pd-test" # 'preprecess', trained model, evaluation log
        # preprocess_save_path = os.path.join(EXPERIMENTAL_NAME, 'model', EXPERIMENTAL_NAME[2:], "standardscaler.pkl")
        preprocess_save_path = None

        extension = "nii"
        dataDim = "3D" # 3D
        modeling = "3D"
        view = "axial"

        lr_rate = 0.00005
        num_classes = 2
        #num_epochs = 1000
        num_epochs = 200
        input_shape = (64,64,64) # (D, H, W)
        channel_size = 1
        batch_size = 32
        #bbox = (50, 50, 50 + 300, 50 + 300)
        # additional feature data
        # excel_filepath = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_10\\FBB_10_final_clinical_dataset.xlsx"
        # id_column_name = "PatientID(new)"
        # columns = ["PatientID(new)", "Sex", "Age", "Education", "MMSE", "CDR", "CDR-SOB", "GDS", "SGDepS"]

        # conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, num_epochs = num_epochs, root_dir=".\\_test_clinical_info_epoch3000", model_save_dir="models"
        #              , result_save_dir="results", batch_size=64, cv_file=None, numAdditionalFeatures=len(columns)-1)
        conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, input_shape = input_shape , channel_size=channel_size, num_epochs=num_epochs,
                                  root_dir=EXPERIMENTAL_NAME, model_save_dir="models"
                     , result_save_dir="results", batch_size=batch_size, cv_file=None, numAdditionalFeatures=None, model_reload_path=None)


        # preprocessing before modeling : resize, channel
        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=input_shape)
        resized_scaled_datas = [imgip.interpolateImage(img) for img in datas]
        if resized_scaled_datas[0].shape[-1] != 1 or resized_scaled_datas[0].shape[-1] != 3:
            resized_scaled_datas = np.array(resized_scaled_datas)[:, :, :, :, None]

        # Split data into training and testing
        X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test = train_test_split(resized_scaled_datas, labels, data_filenames, train_size=0.70, random_state=1)
        print("X_train.shape", X_train.shape) # X_train.shape (277, 64, 64, 64, 1)
        print("X_test.shape", X_test.shape) # X_test.shape (120, 64, 64, 64, 1)
        print("data_filenames_train.shape", data_filenames_train.shape)  # X_test.shape (77, 68, 95, 79)


        # # split data by group label
        # bapl1_nii = scaled_datas[labels == 0]
        # bapl23_nii = scaled_datas[labels == 1]
        # print(bapl1_nii.shape)  # (157, 68, 95, 79)
        # print(bapl23_nii.shape)  # (99, 68, 95, 79)
        #
        # conv3dmodel.load3DImage(data_dir, extension=extension, data_dir_child="labeled",
        #                         dataDim=dataDim, instanceIsOneFile=True, view=view)

        conv3dmodel.updateInput(datas=X_train, labels=y_train, data_filenames=data_filenames_train, label_name=label_name, class_name=class_name)
        # conv3dmodel.train_eval_CV(num_cv=4)

        s, _ = StandardScalingData(X_train, save_path=None, train=True, keep_dim=True)
        _, scaled_X_train = StandardScalingData(X_train, save_path=None, train=False, keep_dim=True, scaler=s)
        _, scaled_X_test = StandardScalingData(X_test, save_path=None, train=False, keep_dim=True, scaler=s)
        conv3dmodel.train_eval(scaled_X_train, y_train, scaled_X_test, y_test, EXPERIMENTAL_NAME, save_model_name=None)
        conv3dmodel.test(scaled_X_test, y_test, EXPERIMENTAL_NAME, data_filenames_test, preprocess_save_path=preprocess_save_path)

        # grad-CAM
        # _preds = conv3dmodel.test(X_test, y_test, EXPERIMENTAL_NAME, data_filenames_test)
        #
        # CAM_save_dir = r'./3D_CNN_apl1vs23_210706_4_raw_data_GradCAM/GradCAM_log'
        # if not os.path.isdir(CAM_save_dir):
        #     os.mkdir(CAM_save_dir)
        # conv3dmodel._visualize_3D_CAM(X_test[:2], y_test[:2], _preds[:2], data_filenames_test[:2], CAM_save_dir, "conv3d_2",
        #                   resize_size=(64, 64, 64), experiment_name_for_preprocess=None, preprocess_save_path=None)


    elif task_mode == "3D_CNN_model_test_with_ft_sel":
        print("Conv3DModel 3D_CNN_model_test_with_VWMT_bon")

        feature_selector = 'Marginal_test'  # 'Marginal_test', 'Lasso', 'Lasso_iter', 'export_mask'


        EXPERIMENTAL_NAME = "./3D_CNN_apl1vs23_210517_with_VWMT_bon_0.01"

        data_dir = r"/media/ubuntu/40d020bc-904f-49b4-ac95-2cfbaec96c2a/2_CN_SN_raw_nii"
        # preprocess_save_path = os.path.join(EXPERIMENTAL_NAME, 'model', EXPERIMENTAL_NAME[2:], "standardscaler.pkl")
        preprocess_save_path = None

        extension = "nii"
        dataDim = "3D" # 3D
        modeling = "3D"
        view = "axial"

        lr_rate = 0.00005
        num_classes = 2
        #num_epochs = 1000
        num_epochs = 100
        input_shape = (64,64,64)
        channel_size = 1
        #bbox = (50, 50, 50 + 300, 50 + 300)
        # additional feature data
        # excel_filepath = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_10\\FBB_10_final_clinical_dataset.xlsx"
        # id_column_name = "PatientID(new)"
        # columns = ["PatientID(new)", "Sex", "Age", "Education", "MMSE", "CDR", "CDR-SOB", "GDS", "SGDepS"]

        # conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, num_epochs = num_epochs, root_dir=".\\_test_clinical_info_epoch3000", model_save_dir="models"
        #              , result_save_dir="results", batch_size=64, cv_file=None, numAdditionalFeatures=len(columns)-1)
        conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, input_shape = input_shape , channel_size=channel_size, num_epochs=num_epochs,
                                  root_dir=EXPERIMENTAL_NAME, model_save_dir="models"
                     , result_save_dir="results", batch_size=16, cv_file=None, numAdditionalFeatures=None, model_reload_path=None)


        # preprocessing before modeling : resize, channel
        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=input_shape)
        resized_scaled_datas = [imgip.interpolateImage(img) for img in scaled_datas]
        if resized_scaled_datas[0].shape[-1] != 1 or resized_scaled_datas[0].shape[-1] != 3:
            resized_scaled_datas = np.array(resized_scaled_datas)[:, :, :, :, None]

        # Split data into training and testing
        X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test = train_test_split(resized_scaled_datas, labels, data_filenames, train_size=0.70, random_state=1)
        print("X_train.shape", X_train.shape) # X_train.shape (179, 68, 95, 79)
        print("X_test.shape", X_test.shape) # X_test.shape (77, 68, 95, 79)
        print("data_filenames_train.shape", data_filenames_train.shape)  # X_test.shape (77, 68, 95, 79)


        # Feature selection
        if feature_selector == 'Marginal_test':
            p_val_adjust_method = 'bonferroni'  # 'fdr_bh'  # 'bonferroni'
            output_save_dirname = "210517_VWMT_bonferroni_alpha_0.001_std_scaler_test"
            output_save_dir = os.path.join(
                r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210517\VWMT',
                output_save_dirname)

            alpha = 0.01  # 0.05, 0.01, 0.001

            bapl1_nii = X_train[y_train == 0]
            bapl23_nii = X_train[y_train == 1]
            print("#bapl1_nii in train", bapl1_nii.shape)
            print("#bapl23_nii in train", bapl23_nii.shape)
            # calculating p-value matrix
            m_testor = MarginalTest()
            statistics = m_testor.create_p_matrix(bapl1_nii, bapl23_nii, type_test='ttest_ind')

            # non-adjust
            # pval_map = statistics[1]
            # pval_mask = np.zeros(pval_map.shape)
            #
            # pval_mask[pval_map >= alpha] = 2**10
            # pval_mask[pval_map<alpha] = 1
            # pval_mask[pval_mask == 2**10] = 0

            # nii_io.show_one_img_v3(pval_map, is2D=False, cmap=plt.get_cmap('gray'))
            # nii_io.show_one_img_v3(pval_mask, is2D=False, cmap=plt.get_cmap('gray'))

            # adjust
            statistics = m_testor.fdr_masking(statistics[1], alpha=alpha, method=p_val_adjust_method)
            adjust_pval_map = statistics[1]
            adjust_pval_mask = statistics[0]
            # nii_io.show_one_img_v3(pval_map, is2D=False, cmap=plt.get_cmap('gray'))
            # nii_io.show_one_img_v3(adjust_pval_mask, is2D=False, cmap=plt.get_cmap('gray'))

            # # calculate num_sel according to index of slice
            # num_sel_on_target_slice = np.zeros(
            #     [2, 68])  # FL TL  등 target region sel은 최대이면서 nonspecific region selection은 최소화 하는 alpha 탐색으로 만든 mask
            #
            # # for slice_ind in range(len(pval_mask)):
            # #     target_slice = pval_mask[slice_ind]
            # #     num_sel = target_slice[target_slice==1]
            # #     num_sel_on_target_slice[0][slice_ind] = len(num_sel)
            # #
            # for slice_ind in range(len(adjust_pval_mask)):
            #     target_slice = adjust_pval_mask[slice_ind]
            #     num_sel = target_slice[target_slice == 1]
            #     num_sel_on_target_slice[1][slice_ind] = len(num_sel)
            # #
            # df = pd.DataFrame({'NAPMF_p_0.0001_mask':num_sel_on_target_slice[0,:], 'VWMT_p_0.001_mask':num_sel_on_target_slice[1,:]})
            # df.to_excel(r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\VWMT_bh\VWMT_p0.001_slice_num_sel.xlsx')

            # # atlas-based feature observation
            # bin_target_region_filepath_list = [
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Frontal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Parietal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Posterior_Cingulate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Temporal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Cerebellum_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Hammers_WholeBrain_2mm_79_95_68.nii']
            #
            # target_value_list = [1, 1, 1, 1, 1, 1, 0]
            # # pval_mask_vals, region_name = atlas_based_eval_pval_mask(pval_mask, bin_target_region_filepath_list,
            # #                                          target_value_list=target_value_list)
            #
            # adjust_pval_mask_vals, region_name = atlas_based_eval_pval_mask(adjust_pval_mask, bin_target_region_filepath_list,
            #                                          target_value_list=target_value_list)
            # df = pd.DataFrame({'VWMT_p_0.001_mask': adjust_pval_mask_vals,
            #                    'region_name':region_name})
            #
            # df.to_excel(
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\VWMT_bh\VWMT_p0.001_atlas_num_sel.xlsx')


        # modeling
        masked_X_train = X_train * adjust_pval_mask
        masked_X_test = X_test * adjust_pval_mask

        conv3dmodel.updateInput(datas=masked_X_train, labels=y_train, data_filenames=data_filenames_train, label_name=label_name, class_name=class_name)
        # conv3dmodel.train_eval_CV(num_cv=4)
        conv3dmodel.train_eval(masked_X_train, y_train, masked_X_test, y_test, EXPERIMENTAL_NAME, save_model_name=None)

        s, _ = StandardScalingData(masked_X_train, save_path=None, train=True, keep_dim=True)
        _, scaled_X_test = StandardScalingData(masked_X_test, save_path=None, train=False, keep_dim=True, scaler=s)
        conv3dmodel.test(scaled_X_test, y_test, EXPERIMENTAL_NAME, data_filenames_test, preprocess_save_path=preprocess_save_path)



    elif task_mode == "DNN_experiment":
        print("Conv3DModel 3D_CNN_model_sel")

        EXPERIMENTAL_NAME = "./3D_CNN_apl1vs23_210516"

        data_dir = r"C:\Users\hkang\PycharmProjects\datas\ADNI\arranged_imgs_sylee_nii_hk\2_CN_SN_raw_nii"
        extension = "nii"
        dataDim = "3D" # 3D
        modeling = "3D"
        view = "axial"

        lr_rate = 0.00005
        dropout_rate = 0.1
        batch_size=128

        num_classes = 2
        #num_epochs = 1000
        num_epochs = 200
        input_shape = (64,64,64)
        channel_size = 1

        hidden_units = [32, 32]

        #bbox = (50, 50, 50 + 300, 50 + 300)
        # additional feature data
        # excel_filepath = "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_10\\FBB_10_final_clinical_dataset.xlsx"
        # id_column_name = "PatientID(new)"
        # columns = ["PatientID(new)", "Sex", "Age", "Education", "MMSE", "CDR", "CDR-SOB", "GDS", "SGDepS"]

        # conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, num_epochs = num_epochs, root_dir=".\\_test_clinical_info_epoch3000", model_save_dir="models"
        #              , result_save_dir="results", batch_size=64, cv_file=None, numAdditionalFeatures=len(columns)-1)
        conv3dmodel = Conv3DModel(lr_rate=lr_rate, num_classes=num_classes, input_shape = input_shape , channel_size=channel_size, num_epochs=num_epochs,
                                  root_dir=EXPERIMENTAL_NAME, model_save_dir="models"
                     , result_save_dir="results", batch_size=16, cv_file=None, numAdditionalFeatures=None, model_reload_path=None)

        # preprocessing before modeling : resize, channel
        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=input_shape)
        resized_scaled_datas = [imgip.interpolateImage(img) for img in scaled_datas]
        if resized_scaled_datas[0].shape[-1] != 1 or resized_scaled_datas[0].shape[-1] != 3:
            resized_scaled_datas = np.array(resized_scaled_datas)[:, :, :, :, None]

        # Split data into training and testing
        X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test = train_test_split(resized_scaled_datas, labels, data_filenames, train_size=0.70, random_state=1)
        print("X_train.shape", X_train.shape) # X_train.shape (179, 68, 95, 79)
        print("X_test.shape", X_test.shape) # X_test.shape (77, 68, 95, 79)
        print("data_filenames_train.shape", data_filenames_train.shape)  # X_test.shape (77, 68, 95, 79)


        # # split data by group label
        # bapl1_nii = scaled_datas[labels == 0]
        # bapl23_nii = scaled_datas[labels == 1]
        # print(bapl1_nii.shape)  # (157, 68, 95, 79)
        # print(bapl23_nii.shape)  # (99, 68, 95, 79)
        #
        # conv3dmodel.load3DImage(data_dir, extension=extension, data_dir_child="labeled",
        #                         dataDim=dataDim, instanceIsOneFile=True, view=view)
        conv3dmodel.updateInput(datas=X_train, labels=y_train, data_filenames=data_filenames_train, label_name=label_name, class_name=class_name)
        # conv3dmodel.train_eval_CV(num_cv=4)
        _, _, val_acc, val_loss = conv3dmodel.train_eval(X_train, y_train, X_test, y_test,
                                                  experiment_name="experiment_CV_" + str(ind),
                                                  save_model_name=None)

        # datas = nii_io._3D_data
        # labels = nii_io._label
        # data_filenames = nii_io._3D_data_filename
        # label_name = nii_io._label_name
        # class_name = nii_io._class_name
        # print(np.array(conv3dmodel._3D_data).shape) # (213, 68, 95, 79, 1)
        # print(np.array(conv3dmodel._3D_data_filename).shape) # (256,)
        # print(np.array(conv3dmodel._label).shape) # (256,)
        # print(np.array(conv3dmodel._label_name).shape) # (256,)
        # print(np.array(conv3dmodel._class_name)) # ['0_bapl1' '1_bapl23']
    elif task_mode == "SVM":
        print("SVM")

        feature_selector = 'Marginal_test'  # 'Marginal_test', 'Lasso', 'Lasso_iter', 'export_mask'


        # Split data into training and testing
        X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test = train_test_split(
            scaled_datas, labels, data_filenames, train_size=0.70, random_state=1)
        print("X_train.shape", X_train.shape)  # X_train.shape (179, 68, 95, 79)
        print("X_test.shape", X_test.shape)  # X_test.shape (77, 68, 95, 79)
        print("data_filenames_train.shape", data_filenames_train.shape)  # X_test.shape (77, 68, 95, 79)
        print("X_train.min(), X_train.max()", X_train.min(), X_train.max()) # X_train.min(), X_train.max() 0.0 255.0

        # # preprocessing
        # s, X_train = StandardScalingData(X_train, save_path=None, keep_dim=True, train=True, scaler=None)
        #
        # _, X_test = StandardScalingData(X_test, save_path=None, keep_dim=True, train=False, scaler=s)

        # Feature selection
        if feature_selector == 'Marginal_test':

            p_val_adjust_method = 'bonferroni' # 'fdr_bh'  # 'bonferroni'
            output_save_dirname = "210517_VWMT_bonferroni_alpha_0.001_std_scaler_test"
            output_save_dir = os.path.join(
                r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210517\VWMT',
                output_save_dirname)

            alpha = 0.001 # 0.05, 0.01, 0.001

            bapl1_nii = X_train[y_train==0]
            bapl23_nii = X_train[y_train == 1]
            print("#bapl1_nii in train", bapl1_nii.shape)
            print("#bapl23_nii in train", bapl23_nii.shape)
            # calculating p-value matrix
            m_testor = MarginalTest()
            statistics = m_testor.create_p_matrix(bapl1_nii, bapl23_nii, type_test='ttest_ind')

            # non-adjust
            # pval_map = statistics[1]
            # pval_mask = np.zeros(pval_map.shape)
            #
            # pval_mask[pval_map >= alpha] = 2**10
            # pval_mask[pval_map<alpha] = 1
            # pval_mask[pval_mask == 2**10] = 0

            # nii_io.show_one_img_v3(pval_map, is2D=False, cmap=plt.get_cmap('gray'))
            # nii_io.show_one_img_v3(pval_mask, is2D=False, cmap=plt.get_cmap('gray'))

            # adjust
            statistics = m_testor.fdr_masking(statistics[1], alpha=alpha, method=p_val_adjust_method)
            adjust_pval_map = statistics[1]
            adjust_pval_mask = statistics[0]
            # nii_io.show_one_img_v3(pval_map, is2D=False, cmap=plt.get_cmap('gray'))
            # nii_io.show_one_img_v3(adjust_pval_mask, is2D=False, cmap=plt.get_cmap('gray'))

            # # calculate num_sel according to index of slice
            # num_sel_on_target_slice = np.zeros(
            #     [2, 68])  # FL TL  등 target region sel은 최대이면서 nonspecific region selection은 최소화 하는 alpha 탐색으로 만든 mask
            #
            # # for slice_ind in range(len(pval_mask)):
            # #     target_slice = pval_mask[slice_ind]
            # #     num_sel = target_slice[target_slice==1]
            # #     num_sel_on_target_slice[0][slice_ind] = len(num_sel)
            # #
            # for slice_ind in range(len(adjust_pval_mask)):
            #     target_slice = adjust_pval_mask[slice_ind]
            #     num_sel = target_slice[target_slice == 1]
            #     num_sel_on_target_slice[1][slice_ind] = len(num_sel)
            # #
            # df = pd.DataFrame({'NAPMF_p_0.0001_mask':num_sel_on_target_slice[0,:], 'VWMT_p_0.001_mask':num_sel_on_target_slice[1,:]})
            # df.to_excel(r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\VWMT_bh\VWMT_p0.001_slice_num_sel.xlsx')

            # # atlas-based feature observation
            # bin_target_region_filepath_list = [
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Frontal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Parietal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Posterior_Cingulate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Temporal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Cerebellum_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Hammers_WholeBrain_2mm_79_95_68.nii']
            #
            # target_value_list = [1, 1, 1, 1, 1, 1, 0]
            # # pval_mask_vals, region_name = atlas_based_eval_pval_mask(pval_mask, bin_target_region_filepath_list,
            # #                                          target_value_list=target_value_list)
            #
            # adjust_pval_mask_vals, region_name = atlas_based_eval_pval_mask(adjust_pval_mask, bin_target_region_filepath_list,
            #                                          target_value_list=target_value_list)
            # df = pd.DataFrame({'VWMT_p_0.001_mask': adjust_pval_mask_vals,
            #                    'region_name':region_name})
            #
            # df.to_excel(
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\VWMT_bh\VWMT_p0.001_atlas_num_sel.xlsx')

            # modeling
            # modeling
            svm = SVM(kernel='linear', C=10.0, gamma='auto', seed=1, probability=True)
            # raw image
            # flatten_X_train = X_train.reshape([len(X_train), -1])
            # flatten_X_test = X_test.reshape([len(X_test), -1])

            masked_X_train = X_train * adjust_pval_mask
            masked_X_test = X_test * adjust_pval_mask

            # flatten_X_train = masked_X_train.reshape([len(X_train), -1])
            # flatten_X_test = masked_X_test.reshape([len(X_test), -1])
            # flatten_loaded_mask = mask_map.flatten().tolist()*len(X_train)
            #
            # flatten_X_train = flatten_X_train[flatten_loaded_mask==1]
            # flatten_X_test = flatten_X_test[flatten_loaded_mask == 1]

            flatten_X_train = np.array([masked_X[adjust_pval_mask == 1] for masked_X in masked_X_train])
            flatten_X_test = np.array([masked_X[adjust_pval_mask == 1] for masked_X in masked_X_test])

            print("flatten_X_train.shape", flatten_X_train.shape)
            print("flatten_X_test.shape", flatten_X_test.shape)
            _, _, val_accuracy, val_f1_score = svm.train_eval(flatten_X_train, y_train, flatten_X_test, y_test,
                                                              X_train_filename=data_filenames_train,
                                                              X_test_filename=data_filenames_test,
                                                              output_save_dir=output_save_dir)
            print("val_accuracy", val_accuracy)
            print("val_f1_score", val_f1_score)

        elif feature_selector == 'Lasso':
            lasso_ref = Lasso()

            output_save_dirname = "210517_svm_linear_C10.0_sklearn_L1_C_10_test_ver2"

            output_save_dir = os.path.join(
                r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210517\export_mask_Lasso',
                output_save_dirname)

            C = 10.0 # lambda : 0.01

            #data_shape = scaled_datas.shape
            data_shape = X_train.shape

            # flatten data
            # flatten_datas = scaled_datas.reshape([data_shape[0], data_shape[1] * data_shape[2] * data_shape[3]])
            flatten_datas = X_train.reshape([data_shape[0], data_shape[1] * data_shape[2] * data_shape[3]])

            print("flatten_datas.shape", flatten_datas.shape)
            # lasso_ref.train_lasso_eval(X_train=flatten_datas, y_train=labels, X_test=None, y_test=None, C=C)

            lasso_ref.train_lasso_eval(X_train=flatten_datas, y_train=y_train, X_test=None, y_test=None, C=C)


            betas = lasso_ref.get_coef()
            print("betas.shape", betas.shape)
            reshaped_betas = betas.reshape(data_shape[1], data_shape[2], data_shape[3])
            print("test np.unique(reshaped_betas)", np.unique(reshaped_betas).shape, reshaped_betas.min(),
                  reshaped_betas.max())

            mask_map = np.zeros(reshaped_betas.shape)
            mask_map[reshaped_betas == 0] = 0
            mask_map[reshaped_betas != 0] = 1

            print("test np.unique(mask_map)", np.unique(mask_map).shape)

            # # save mask
            # # print("np.array(mask_map).shape", np.array(mask_map).shape)
            # mask_savepath = r"C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\results_210516\lasso_C100_mask.npy"
            # with open(mask_savepath, "wb") as f:
            #     np.save(f, np.array(mask_map))
            # nii_io.show_one_img_v3(mask_map, is2D=False, cmap=plt.get_cmap('gray'))
            #
            # # load mask
            # mask_savepath = r"C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\results_210516\lasso_C100_mask.npy"
            # with open(mask_savepath, "rb") as f:
            #     loaded_mask_map = np.load(f)
            # nii_io.show_one_img_v3(loaded_mask_map, is2D=False, cmap=plt.get_cmap('gray'))

            # # calculate num_sel according to index of slice
            # num_sel_on_target_slice = np.zeros(
            #     [1, 68])  # FL TL  등 target region sel은 최대이면서 nonspecific region selection은 최소화 하는 alpha 탐색으로 만든 mask
            #
            # for slice_ind in range(len(mask_map)):
            #     target_slice = mask_map[slice_ind]
            #     num_sel = target_slice[target_slice == 1]
            #     num_sel_on_target_slice[0][slice_ind] = len(num_sel)
            #
            # df = pd.DataFrame({'L1_C_100.0': num_sel_on_target_slice[0, :]})
            # df.to_excel(
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\L1\L1_C_100.0_slice_num_sel.xlsx')
            #
            # # atlas-based feature observation
            # bin_target_region_filepath_list = [
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Frontal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Parietal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Posterior_Cingulate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Temporal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Cerebellum_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Hammers_WholeBrain_2mm_79_95_68.nii']
            #
            # target_value_list = [1, 1, 1, 1, 1, 1, 0]
            # pval_mask_vals, region_name = atlas_based_eval_pval_mask(mask_map, bin_target_region_filepath_list,
            #                                          target_value_list=target_value_list)
            #
            #
            # df = pd.DataFrame({'C_100.0_mask': pval_mask_vals,
            #                    'region_name':region_name})
            #
            # df.to_excel(
            #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210513\L1\L1_C_100.0_atlas_num_sel.xlsx')

            # modeling
            svm = SVM(kernel='linear', C=10.0, gamma='auto', seed=1, probability=True)
            # raw image
            # flatten_X_train = X_train.reshape([len(X_train), -1])
            # flatten_X_test = X_test.reshape([len(X_test), -1])

            masked_X_train = X_train * mask_map
            masked_X_test = X_test * mask_map

            # flatten_X_train = masked_X_train.reshape([len(X_train), -1])
            # flatten_X_test = masked_X_test.reshape([len(X_test), -1])
            # flatten_loaded_mask = mask_map.flatten().tolist()*len(X_train)
            #
            # flatten_X_train = flatten_X_train[flatten_loaded_mask==1]
            # flatten_X_test = flatten_X_test[flatten_loaded_mask == 1]

            flatten_X_train = np.array([masked_X[mask_map == 1] for masked_X in masked_X_train])
            flatten_X_test = np.array([masked_X[mask_map == 1] for masked_X in masked_X_test])

            print("flatten_X_train.shape", flatten_X_train.shape)
            print("flatten_X_test.shape", flatten_X_test.shape)
            _, _, val_accuracy, val_f1_score = svm.train_eval(flatten_X_train, y_train, flatten_X_test, y_test,
                                                              X_train_filename=data_filenames_train,
                                                              X_test_filename=data_filenames_test,
                                                              output_save_dir=output_save_dir)
            print("val_accuracy", val_accuracy)
            print("val_f1_score", val_f1_score)

        elif feature_selector == 'Lasso_iter':
            lasso_ref = Lasso()

            # C = 0.001
            C_max = 1.0
            C_min = 0.0001
            step = 20
            # increment = (C_max - C_min) / step
            # iter_C_under_one = np.arange(C_min, C_max, increment)

            # C_max = 201
            # C_min = 1
            # step = 200
            increment = (C_max - C_min) / step
            iter_C_over_one = np.arange(C_min, C_max, increment)
            iter_C = iter_C_over_one
            print("iter_C", len(iter_C))
            #iter_C = np.concatenate([iter_C_under_one, iter_C_over_one])

            df_list_to_bind_by_slice= []
            val_accuracy_list = []
            val_f1_score_list = []
            for iter_C_elem in iter_C:

                data_shape = datas.shape
                # flatten data
                flatten_datas = datas.reshape([data_shape[0], data_shape[1] * data_shape[2] * data_shape[3]])
                print("flatten_datas.shape", flatten_datas.shape)
                lasso_ref.train_lasso_eval(X_train=flatten_datas, y_train=labels, X_test=None, y_test=None, C=iter_C_elem)

                betas = lasso_ref.get_coef()
                print("betas.shape", betas.shape)
                reshaped_betas = betas.reshape(data_shape[1], data_shape[2], data_shape[3])
                print("test np.unique(reshaped_betas)", np.unique(reshaped_betas).shape, reshaped_betas.min(),
                      reshaped_betas.max())

                mask_map = np.zeros(reshaped_betas.shape)
                mask_map[reshaped_betas == 0] = 0
                mask_map[reshaped_betas != 0] = 1

                #print("test np.unique(mask_map)", np.unique(mask_map).shape)
                #nii_io.show_one_img_v3(mask_map, is2D=False, cmap=plt.get_cmap('gray'))

                # # calculate num_sel according to index of slice
                # num_sel_on_target_slice = np.zeros(
                #     [1, 68])  # FL TL  등 target region sel은 최대이면서 nonspecific region selection은 최소화 하는 alpha 탐색으로 만든 mask
                #
                # for slice_ind in range(len(mask_map)):
                #     target_slice = mask_map[slice_ind]
                #     num_sel = target_slice[target_slice == 1]
                #     num_sel_on_target_slice[0][slice_ind] = len(num_sel)
                #
                # df = pd.DataFrame({f'L1_C_{iter_C_elem}_mask': num_sel_on_target_slice[0, :]})
                # df_list_to_bind_by_slice.append(df)


                # # atlas-based feature observation
                # bin_target_region_filepath_list = [
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Frontal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Parietal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Posterior_Cingulate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\1_specific_region\Temporal_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Cerebellum_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii',
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\atlas_based_feature_dist_observation\Hammers atlas\2_non_specific_region\Hammers_WholeBrain_2mm_79_95_68.nii']
                #
                # target_value_list = [1, 1, 1, 1, 1, 1, 0]
                # pval_mask_vals, region_name = atlas_based_eval_pval_mask(mask_map, bin_target_region_filepath_list,
                #                                          target_value_list=target_value_list)
                #
                #
                # df = pd.DataFrame({'C_100.0_mask': pval_mask_vals,
                #                    'region_name':region_name})
                #
                # df.to_excel(
                #     r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\result_210507\atlas_based_L1_C100_num_sel.xlsx')



                # X_train, X_test, y_train, y_test, data_filenames_train, data_filenames_test
                # label_name, class_name

                # # SVM : model selection
                # svm = SVM(kernel='poly', C=0.1, gamma='auto', seed = 1, probability=True)
                # flatten_X_train = X_train.reshape([len(X_train), -1])
                #
                # svm.LoadData(datas=flatten_X_train, labels=y_train, data_filenames=data_filenames_train, label_name=label_name,
                #              class_name=class_name)
                # svm.train_eval_CV(num_cv=4)

                output_save_dirname = "210507_svm_linear_C10.0_L1_C{iter_C_elem}_test"
                output_save_dir = os.path.join(r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210510\L1_iter_C', output_save_dirname)
                # output_save_dir=None
                # SVM : model test

                svm = SVM(kernel='linear', C=10.0, gamma='auto', seed = 1, probability=True)

                masked_X_train = X_train*mask_map
                masked_X_test = X_test * mask_map


                flatten_X_train = masked_X_train.reshape([len(X_train), -1])
                flatten_X_test = masked_X_test.reshape([len(X_test), -1])


                _, _, val_accuracy, val_f1_score = svm.train_eval(flatten_X_train, y_train, flatten_X_test, y_test,
                               X_train_filename=data_filenames_train, X_test_filename=data_filenames_test,
                               output_save_dir=output_save_dir)
                val_accuracy_list.append(val_accuracy)
                val_f1_score_list.append(val_f1_score)

            total_df = pd.DataFrame({"val_accuracy_list": val_accuracy_list, "val_f1_score_list": val_f1_score_list})
            total_df.to_excel(
                r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210510\L1_iter_C\total_performance.xlsx')



        elif feature_selector == 'export_mask':

            mask_path = r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\marginal_test_results\210516-colab실험데이터\210516\SCAD\scad_0.01_mask.npy'
            with open(mask_path, 'rb') as f:
                loaded_mask = np.load(f)

            output_save_dirname = "210517_svm_linear_C10.0_SCAD_lambda_0.01_test"
            output_save_dir = os.path.join(
                r'C:\Users\hkang\PycharmProjects\FeatureSelectionAnalysis\model_generalization\210517\export_mask_Lasso',
                output_save_dirname)

            # output_save_dir=None
            # SVM : model test

            svm = SVM(kernel='linear', C=10.0, gamma='auto', seed=1, probability=True)
            # raw image
            # flatten_X_train = X_train.reshape([len(X_train), -1])
            # flatten_X_test = X_test.reshape([len(X_test), -1])

            masked_X_train = X_train * loaded_mask
            masked_X_test = X_test * loaded_mask

            # flatten_X_train = masked_X_train.reshape([len(X_train), -1])
            # flatten_X_test = masked_X_test.reshape([len(X_test), -1])
            # flatten_loaded_mask = loaded_mask.flatten()
            # flatten_loaded_mask = flatten_loaded_mask.tolist()*

            # flatten_X_train = flatten_X_train[flatten_loaded_mask==1]
            # flatten_X_test = flatten_X_test[flatten_loaded_mask == 1]

            flatten_X_train = np.array([masked_X[loaded_mask==1] for masked_X in masked_X_train])
            flatten_X_test = np.array([masked_X[loaded_mask==1] for masked_X in masked_X_test])

            print("flatten_X_train.shape", flatten_X_train.shape)
            print("flatten_X_test.shape", flatten_X_test.shape)
            _, _, val_accuracy, val_f1_score = svm.train_eval(flatten_X_train, y_train, flatten_X_test, y_test,
                                                              X_train_filename=data_filenames_train,
                                                              X_test_filename=data_filenames_test,
                                                              output_save_dir=output_save_dir)
            print("val_accuracy", val_accuracy)
            print("val_f1_score", val_f1_score)
