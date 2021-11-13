import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#from MaskImage import mask_image


from ImageDataIO import ImageDataIO
from DataIO import NIIDataIO
from DistributionViewer import DistributionViewer

def load_Dydata(data_path):
    X_data = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_X_data.npy'))
    y_label = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_y_label.npy'))
    X_pid = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_X_pid.npy'))
    print("X_data.shape", X_data.shape)
    print("y_label.shape", y_label.shape)
    print("X_pid.shape", X_pid.shape)
    return X_data, y_label, X_pid

def load_Stdata(data_path):
    X_data = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_X_data.npy'))
    y_label = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_y_label.npy'))
    X_pid = np.load(
        os.path.join(data_path, 'ubuntu_DyFBB_X_pid.npy'))
    print("X_data.shape", X_data.shape)
    print("y_label.shape", y_label.shape)
    print("X_pid.shape", X_pid.shape)
    return X_data, y_label, X_pid

def get_reduced_feature_with_pca(target_data, n_components, pca=None):
    target_data = np.array(target_data)
    target_shape = target_data.shape

    flatten_data = np.array([_x.flatten() for _x in target_data])
    if pca is None :
        _pca = PCA(n_components=n_components)
        embedded_data = _pca.fit_transform(flatten_data)
    else :
        _pca = pca
        embedded_data = _pca.transform(flatten_data)


    return embedded_data, _pca.explained_variance_ratio_, _pca

def get_2d_feature_with_tsne(target_data):
    target_data = np.array(target_data)
    target_shape = target_data.shape

    flatten_data = np.array([_x.flatten() for _x in target_data])
    tsne = TSNE(n_components=2)
    embedded_data = tsne.fit_transform(flatten_data)
    return embedded_data

def concatenate_two_longitudinal_features(feature_a_list, feature_b_list, id_a_list, id_b_list, label_a_list):
    """

    :param feature_a_list: (N_a, T_a, F)
    :param feature_b_list: (N_b, T_b, F)
    :param id_a: (N_a, )
    :param id_b: (N_b, )
    :return: (numOfIntersection(N_a, N_b), T_a+T_b, F)
    """
    feature_a_list = np.array(feature_a_list)
    feature_b_list = np.array(feature_b_list)
    concatenated_feature_list = []
    concatenated_id_list = []
    concatenated_label_list = []
    for index_id_a, _id_a in enumerate(id_a_list):
        if _id_a in id_b_list:
            index_id_b = np.where(id_b_list == _id_a)[0][0]
            feature_b = feature_b_list[index_id_b]
            feature_a = feature_a_list[index_id_a]
            concat_feature_a_b = np.concatenate([feature_a, feature_b], axis=0)
            concatenated_feature_list.append(concat_feature_a_b)
            concatenated_id_list.append(_id_a)
            concatenated_label_list.append(label_a_list[index_id_a])
    return np.array(concatenated_feature_list), np.array(concatenated_label_list), np.array(concatenated_id_list)

if __name__ == "__main__":
    print("check distribution")
    mode = "st" # # "st" # "dy" # "both"

    if mode=="dy":
        print("mode", mode)

        # load original files
        # load_DyFBB_img_data_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/2_StFBB_img_data_4label'
        # resize_size = (32, 32, 32)
        # X_data, y_label, X_pid = load_dynamic_img_data(load_DyFBB_img_data_path, lenOfTime=1, resize_size=resize_size)
        #
        # # save data
        # np.save(os.path.join(r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/3_StFBB_img_data_4label_32_32_32_200827', 'ubuntu_DyFBB_X_data.npy'), X_data)
        # np.save(
        #     os.path.join(r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/3_StFBB_img_data_4label_32_32_32_200827', 'ubuntu_DyFBB_y_label.npy'),
        #     y_label)
        # np.save(os.path.join(r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/3_StFBB_img_data_4label_32_32_32_200827', 'ubuntu_DyFBB_X_pid.npy'),
        #         X_pid)

        filepath = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/1_DyFBB_img_data/0_TEST'
        X_data, X_pid = load_dynamic_img(filepath , lenOfTime=27,
                                                         resize_size=resize_size, channel_size=1)

        print("X_data.shape", X_data.shape) # (3, 27, 32, 32, 32)
        #print("y_label.shape", y_label.shape)
        print("X_pid.shape", X_pid.shape)
        input_shape = X_data.shape

        idio = ImageDataIO(extention="nii", dataDim="3D", instanceIsOneFile=True, modeling="3D", view="axial")
        idio.show_one_img_v2(X_data[0][8], is2D=False, cmap=plt.get_cmap('gray'))

        pet_data_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/3_DyFBB_img_data_4label_32_32_32_200827'

        # load Dynamic PET data
        X_data, y_label, X_pid = load_Dydata(pet_data_path)
        X_data = np.array(X_data)
        figure_save_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/experimental_results/distribution/dyPET'



        for ind in range(27):
            # if ind < 8:
            #     continue
            specific_time_id = ind # 5
            print("debug, X_data.shape", X_data.shape) #

            tmp_X_data = X_data[:,specific_time_id,:,:,:] # (264, 32, 32, 32)
            # check image
            #idio.show_one_img_v2(tmp_X_data[0], is2D=False, cmap=plt.get_cmap('gray'))
            # masking : skull
            mask_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/4_mni_mask/Rorden'
            mask_filename = r'sum_grey_white_79_95_68.nii'
            mask_filepath = os.path.join(mask_path, mask_filename)
            #tmp_X_data = mask_image(tmp_X_data, mask_filepath)


            # masking : specific regional mask
            mask_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/4_mni_mask'
            #mask_filename = r'Caudate_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii'
            mask_filename = r'Occipital_Lobe_Hammers_mith_atlas_n30r83_SPM5_2mm_79_95_68.nii'

            #mask_filepath = os.path.join(mask_path, mask_filename)
            #masked_X_data = mask_image(tmp_X_data, mask_filepath)
            #print("debug, masked_X_data.shape", masked_X_data.shape)

            #tmp_X_data = np.array([tmp_X_data, masked_X_data])
            #tmp_X_data = np.transpose(tmp_X_data, axes=[1, 2, 3, 4, 0])
            #print("debug, tmp_X_data.shape", tmp_X_data.shape)



            title = str(ind) + " early dynamic PET distribution"
            picker = 5
            color_list = "rbgk"

            # feature extraction with PCA
            pca_embedded_X_data, explained_variance_ratios, _ = get_reduced_feature_with_pca(tmp_X_data, n_components=256)
            print("explained_variance_ratios", explained_variance_ratios, np.sum(explained_variance_ratios))

            # feature extraction with tsne
            embedded_X = get_2d_feature_with_tsne(pca_embedded_X_data)
            dis_view = DistributionViewer(embedded_X, y_label, title, picker, color_list)
            # #dis_view.show_2d_features_dist(origin_X=origin_X, type='plot', mu=xs.astype(np.str), std=ys.astype(np.str))

            plt.show()


            # figure_save_filepath = os.path.join(figure_save_path,
            #                                     str(ind)+'_early_PET_pca' + '%.4f' % np.sum(explained_variance_ratios) + '_tsne_'+str(ind)+'.png')
            #
            # plt.savefig(figure_save_filepath)
            # plt.close('all')
            # plt.clf()

    elif mode=="st":
        print("mode", mode)

        pet_data_path = r'D:\2_FP-CIT_PET_norm_PD\2_Preprocessing_classify\2_Count_Norm'
        # load Static PET data
        #X_data, y_label, X_pid = load_Stdata(pet_data_path)
        nii_io = NIIDataIO()
        nii_io.load_v2(pet_data_path, extension="nii", data_dir_child="labeled",
                       dataDim="3D", instanceIsOneFile=True, channel_size=1, view="axial")

        # check image
        idio = ImageDataIO(extention="nii", dataDim="3D", instanceIsOneFile=True, modeling="3D", view="axial")
        #idio.show_one_img_v2(X_data[0][0], is2D=False, cmap=plt.get_cmap('gray'))
        datas = nii_io._3D_data  # (N, D, W, H), (N, 68, 79, 95)
        labels = nii_io._label  # (N, ) [0, 1, ...], (397, )
        data_filenames = nii_io._3D_data_filename  # (N, ), (397, )

        title = "delay static PET distribution"
        picker = 5
        color_list = "rbgk"

        # feature extraction with PCA
        #pca_embedded_X_data, explained_variance_ratios = get_reduced_feature_with_pca(datas, n_components=2) # 100->78, 256->99.48
        #print("explained_variance_ratios", explained_variance_ratios, np.sum(explained_variance_ratios))

        # feature extraction with tsne
        embedded_X = get_2d_feature_with_tsne(datas)
        dis_view = DistributionViewer(embedded_X, labels, title, picker, color_list)
        dis_view.show_2d_features_dist(origin_X=datas, type='imshow_3d', filename=data_filenames)
        #plt.show()


    elif mode == "both":
        print("mode", mode)

        # load Dynamic PET data
        pet_data_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/2_DyFBB_img_data_4label_32_32_32'
        early_X_data, early_y_label, early_X_pid = load_Dydata(pet_data_path)

        # load Static PET data
        pet_data_path = r'/home/ubuntu/hkang_analysis/cocolab/LSTMClassification/datas/2_StFBB_img_data_4label_32_32_32'
        delay_X_data, delay_y_label, delay_X_pid = load_Stdata(pet_data_path)

        X_data, y_label, X_pid = concatenate_two_longitudinal_features(early_X_data, delay_X_data, early_X_pid,
                                                                       delay_X_pid, early_y_label)
        print("concat_X_data.shape", X_data.shape)
        print("concat_y_label.shape", y_label.shape)
        print("concat_X_pid.shape", X_pid.shape)
        input_shape = X_data.shape

        title = "delay static PET distribution"
        picker = 5
        color_list = "rbgk"

        # feature extraction with PCA
        # pca_embedded_X_data, explained_variance_ratios = get_reduced_feature_with_pca(X_data,
        #                                                                               n_components=256)  # 100->78, 256->99.48
        # print("explained_variance_ratios", explained_variance_ratios, np.sum(explained_variance_ratios))

        # feature extraction with tsne
        embedded_X = get_2d_feature_with_tsne(X_data)
        dis_view = DistributionViewer(embedded_X, y_label, title, picker, color_list)
        # dis_view.show_2d_features_dist(origin_X=origin_X, type='plot', mu=xs.astype(np.str), std=ys.astype(np.str))
        plt.show()

