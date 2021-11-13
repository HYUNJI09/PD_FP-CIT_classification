import os
import sys
import glob
import numpy as np
import pandas as pd
from nibabel import load as nib_load
from scipy import misc
import cv2
from PIL import Image
from keras.utils import to_categorical

#from rnn_model import *

#from img_aug_utils import img_aug_flip_rot
# from cgan import *


def load_nii_data_on_a_plane(data_path, plane):
    """

    :param data_path:
    :param plane: same with mrcro
    :return:
    """
    whole_voxel = nib_load(data_path)
    whole_voxel = np.array(whole_voxel.get_data())
    print("nii shape", whole_voxel.shape) # (W, H, D)

    if plane == "axial" or plane == "transaxial":
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(0, 1)) # (95, 79, 68)
        print("axial nii shape", whole_voxel.shape)  # (W, H, D)
    elif plane == "coronal":
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(1, 2))  #
        whole_voxel = np.rot90(whole_voxel, k=3, axes=(0, 1))
        print("coronal nii shape", whole_voxel.shape)  # (W, H, D)
    elif plane == "saggital":
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(0, 2))  #
        #whole_voxel = np.rot90(whole_voxel, k=2, axes=(1, 2))
        print("saggital nii shape", whole_voxel.shape)  # (W, H, D)
    whole_voxel = whole_voxel.astype(np.uint8)
    whole_voxel = np.transpose(whole_voxel, [2, 0, 1])

    filename = os.path.basename(data_path)
    return np.array(whole_voxel), np.array(filename)

# read nii data
def read_nii_data(data_dir):
    subdir_full_path = [os.path.join(data_dir, file_path) for file_path in os.listdir(data_dir) ]
    nii_imgs, _ = load_nii_slice(subdir_full_path, (14, 50))
    print("loaded nii_data shape", nii_imgs.shape)
    return nii_imgs

#
def read_png_data(data_dir, extention="*.png", key=None):
    #subdir_full_path = [os.path.join(data_dir, file_path) for file_path in os.listdir(data_dir)]
    #subdir_full_path = glob.glob(os.path.join(data_dir, extention))
    subdir_full_path = glob.glob(data_dir+"/"+extention)
    if key is None:
        subdir_full_path.sort(key=lambda x:os.path.basename(x))
    else :
        subdir_full_path.sort(key=key)
    print(subdir_full_path)
    png_imgs, _ = load_png_slice(subdir_full_path) #
    return png_imgs

def loadImgFileFromDir(X_data_path, func=None, sampleSize=None):
    """
    :param X_data_path:
    :return: (S, H, W), S:number of slices (P * 36), H:height, W:width
    """
    child_full_path = [os.path.join(X_data_path, path) for path in os.listdir(X_data_path)]
    child_full_path = np.array(child_full_path)
    if sampleSize is not None:
        child_full_path_ind = np.random.choice(len(child_full_path), sampleSize, replace=False)
        child_full_path = child_full_path[child_full_path_ind]
    data = dict()
    data["image"]=[]
    data["filename"] = []
    # X_data = [] # result as a list of slice samples
    # x_filename = []

    if func is None:
        func = lambda x: x
    for path in child_full_path:
        #print("debug", path)
        #pixels = misc.imread(path)
        img = Image.open(path)
        img = func(img)
        data["image"].append(np.array(img))

        pid_filename = os.path.basename(path) # remove .nii
        data["filename"].append(pid_filename)
    data["image"] = np.array(data["image"])
    data["filename"] = np.array(data["filename"])
    return data

def remove_duplication(whole_data_path_list):
    removed_list = []
    for cls_data_full_path in whole_data_path_list:
        removed_cls_list = []
        for path in cls_data_full_path:
            removed_cls_list.append(path[:-7])
        removed_list.append(np.array(list(set(removed_cls_list))))
    return removed_list

def load_file_data(data_dir):
    abs_data_paths = [os.path.join(data_dir, path) for path in os.listdir(data_dir)]
    abs_data_paths.sort()
    return abs_data_paths

def read_img_with_pil_from_dir(data_dir):
    abs_data_paths = load_file_data(data_dir)
    abs_data_paths.sort()
    imgs = []
    for img_path in abs_data_paths:
        imgs.append(Image.open(img_path))
    return imgs

def read_img_with_pil(data_path_list):
    results = []
    for data_path in data_path_list:
        results.append(Image.open(data_path))
    return results

def get_all_data_full_path(data_path, extention_cond=None, isdir=False):
    whole_data_full_path = []

    class_list = os.listdir(data_path)
    class_list.sort()
    cls_full_path_list = [os.path.join(data_path, cls_) for cls_ in class_list]
    if isdir is True:
        for cls_full_path in cls_full_path_list:
            dir_lists = [os.path.join(cls_full_path, dir_) for dir_ in os.listdir(cls_full_path)]
            whole_data_full_path.append(dir_lists)

    else :
        for cls_full_path in cls_full_path_list:
            data = glob.glob(os.path.join(cls_full_path, extention_cond))
            whole_data_full_path.append(np.array(data))
    return np.array(whole_data_full_path)

# add _index and jpg extention
def generate_3slices_jpg_path(data_path, data_label):
    slices_path_result = []
    slices_label_result = []
    for data_p, label in zip(data_path, data_label):
        data_p_54 = data_p + "_54.jpg"
        data_p_55 = data_p + "_55.jpg"
        data_p_56 = data_p + "_56.jpg"
        slices_path_result.append(data_p_54)
        slices_path_result.append(data_p_55)
        slices_path_result.append(data_p_56)
        slices_label_result.append(label)
        slices_label_result.append(label)
        slices_label_result.append(label)

    return np.array(slices_path_result), np.array(slices_label_result)

# add jpg extention
def generate_1slice_jpg_path(data_path):
    path_result = []
    for data_p in data_path:
        jpg_data_p = data_p + "_55.jpg"
        path_result.append(jpg_data_p)
    return np.array(path_result)


def jpg_list_from_pid_list(data_path_list):
    jpg_res = []
    for data_path in data_path_list:
        pixels = misc.imread(data_path)
        jpg_res.append(np.array(pixels))
    return np.array(jpg_res)

# data augmentation to augment data on each class until each class have same number of dataset
def img_aug_by_cls(img_data, img_label, num_augment):
    class_list = np.unique(img_label)

    augmented_res = []
    augmented_label = []
    for ind in range(len(class_list)):
        # img_data to augment in a class
        cls_img_data = img_data[np.array(img_label)==ind]

        augmented_data = img_aug_flip_rot(cls_img_data, num_augment)
        augmented_res.append(augmented_data)
        augmented_label.append([ind]*num_augment)
    return np.concatenate(augmented_res), np.concatenate(augmented_label)


# data augmentation to augment data on each class until each class have same number of dataset
def img_aug_by_cls_v2(img_data, img_label, num_augment):
    """

    :param img_data: image data to augment
    :param img_label: the label of image data
    :param num_augment: number to augment for each of class, 'num_augment-len(cls_img_data)' is the actual number of augmented data for a class
    :return: original img, original label, augmented img excluding original img and augmented label excluding original label
    """
    class_list = np.unique(img_label)

    origin_res = []
    origin_label = []
    augmented_res = []
    augmented_label = []
    for ind in class_list:
        # img_data to augment in a class
        cls_img_data = img_data[np.array(img_label)==ind]

        # keep original data
        origin_res.append(cls_img_data)
        origin_label.append([ind] * len(cls_img_data))

        # augmented datas to make enough number of data
        real_num_augment = num_augment - len(cls_img_data)
        augmented_data = img_aug_flip_rot(cls_img_data, real_num_augment) # flipping or rotation
        augmented_res.append(augmented_data)
        augmented_label.append([ind]*real_num_augment)

    return np.concatenate(origin_res), np.concatenate(origin_label), np.concatenate(augmented_res), np.concatenate(augmented_label)

def resizing_channel(img_data, resize_size=None, channel_size=None):
    if resize_size is None and channel_size is None:
        print("one of two parameters either resize_size and channel_size have to be obtained")
    if sys.getsizeof(img_data) == 0:
        print("size of img_data object is 0")
        return None
    img_data = np.array(img_data).astype(np.float32)

    # resize
    if resize_size is not None:
        img_data = np.array([cv2.resize(np.array(x), resize_size) for x in img_data]).astype(np.float32)

    img_shape = img_data.shape
    if channel_size is not None and img_shape[-1] is not 1 and img_shape[-1] is not 3:
        img_data = img_data[:, :, :, None]
        img_shape = img_data.shape

    # check channel
    if channel_size == img_shape[-1]:
        return img_data
    if channel_size == 3 and img_shape[-1] == 1:
        img_data = np.array([cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in img_data]).astype(np.float32)
        return img_data
    elif channel_size == 3 and img_shape[-1] == 4:
        #img_arr = color.rgba2rgb(img_arr)
        img_data = np.array([cv2.cvtColor(x, cv2.COLOR_BGRA2BGR) for x in img_data]).astype(np.float32)
        return img_data
    elif channel_size ==1 and img_shape[-1] ==3 :
        img_data = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in img_data]).astype(np.float32)
        return img_data

def resizing_channel_with_PIL(pil_imgs, resize_size, channel_size):
    if sys.getsizeof(pil_imgs) == 0:
        print("size of img_data object is 0")
        return None
    # img_data = np.array(img_data).astype(np.float32)
    # img_shape = img_data.shape

    # resize
    pil_imgs = [x.resize((resize_size)) for x in pil_imgs]
    return pil_imgs
    # check channel
    # if channel_size == img_shape[-1]:
    #     return img_data
    # if channel_size == 3 and img_shape[-1] == 1:
    #     img_data = np.array([cv2.cvtColor(x, cv2.COLOR_GRAY2RGB) for x in img_data]).astype(np.float32)
    #     return img_data
    # elif channel_size == 3 and img_shape[-1] == 4:
    #     #img_arr = color.rgba2rgb(img_arr)
    #     img_data = np.array([cv2.cvtColor(x, cv2.COLOR_BGRA2BGR) for x in img_data]).astype(np.float32)
    #     return img_data
    # elif channel_size ==1 and img_shape[-1] ==3 :
    #     img_data = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in img_data]).astype(np.float32)
    #     return img_data


def resizing_channel_for_RNN(rnn_input, resize_size, channel_size):
    if sys.getsizeof(rnn_input) == 0:
        print("size of img_data object is 0")
        return None

    img_data = np.array(rnn_input[0][0]).astype(np.float32)
    img_shape = img_data.shape
    # if img_shape[-1] is not 1 or img_shape[-1] is not 3:
    #     img_data = img_data[:, :, :, None]
    # img_shape = img_data.shape

    # resize
    img_N = []
    for img_i in rnn_input:
        img_T = []
        for img_t in img_i:
            img_ = np.array(cv2.resize(np.array(img_t), resize_size)).astype(np.float32)

            # check channel
            if channel_size == img_shape[-1]:
                img_T.append(img_)
            if channel_size == 3 and img_shape[-1] == 1:
                img_T.append(np.array(cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)).astype(np.float32))
            elif channel_size == 1 and img_shape[-1] == 3:
                img_T.append(np.array(cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)).astype(np.float32))
        img_N.append(img_T)
    return np.array(img_N).astype(np.float32)

# GAN data augmentation to augment on each class
# have to train gen model_name from only training dataset
# have to return both img_data
def gan_img_aug(gen_path, gen_post_fix, augment_num):

    cgan = CGAN(gen_path, gen_post_fix)
    gen_imgs = cgan.img_aug_by_generator(augment_num) # (augment_num*num_class, 224, 224, 3)
    del cgan

    return gen_imgs

def parse_filename(pid):
    if "_" in pid:
        filename = pid[2:-8]
    elif "x" in pid:
        filename = pid[3:-6]
    return filename

# for bapl1, bapl3
def load_nii_slice(X_data_path, index_interval, y_label=None, slice_label_ref=None):
    """
    :param X_data_path:
    :return: (S, H, W), S:number of slices (P * 36), H:height, W:width
    """
    if slice_label_ref is not None:
        label_ref_df_bapl2 = pd.read_excel(slice_label_ref, sheetname="BAPL2")
        label_ref_df_bapl3 = pd.read_excel(slice_label_ref, sheetname="BAPL3")
    elif slice_label_ref is None:
        slice_label_ref = "/root/PycharmProjects/1903_bapl_gda/Slice grading_final.xlsx"
        label_ref_df_bapl2 = pd.read_excel(slice_label_ref, sheetname="BAPL2")

    X_data = [] # result as a list of slice samples
    x_filename = []

    RCTU_label = [] # (len(X_data), ) # ie. RCTU
    index_label = []  # (len(X_data), 36, ) # ie. to store index, 36 slices information
    for path, y_label_ in zip(X_data_path, y_label):

        whole_voxel = nib_load(path)
        whole_voxel = np.array(whole_voxel.get_data())
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(0, 1))

        whole_voxel = np.array(whole_voxel[:, :, index_interval[0]:index_interval[1]]).astype(np.float32)
        whole_voxel = np.transpose(whole_voxel, [2, 0, 1])
        X_data.append(whole_voxel)

        pid_filename = os.path.basename(path)
        pid_filename = int(parse_filename(pid_filename))
        if slice_label_ref is not None:
            if y_label_==0:
                RCTU_label.append(len(whole_voxel)*[y_label_]) # all RCTU is 0 as RCTU 1
                index_label_ = np.array(list(range(36)))
                index_label.append(index_label_)

            # elif y_label_==1:
            #     disease_label.append(len(whole_voxel)*[y_label])
            #     RCTU_labels = np.array(label_ref_df_bapl2[pid_filename][index_interval[0]:index_interval[1]].tolist()) - 1
            #     index_label_ = np.array(list(range(36)))
            #     index_label.append(index_label_)
            elif y_label_ == 2:

                RCTU_label_ = np.array(label_ref_df_bapl3[pid_filename][index_interval[0]:index_interval[1]].tolist()) - 1
                RCTU_label.append(RCTU_label_)
                index_label_ = np.array(list(range(36)))
                index_label.append(index_label_)
        filename = os.path.basename(path)
        x_filename.append([filename + "_" + str(ind) for ind in range(index_interval[0], index_interval[1])])

    X_data = np.concatenate(X_data)
    x_filename = np.concatenate(x_filename)
    disease_label = np.concatenate(RCTU_label)
    slice_label = np.concatenate(index_label)

    if disease_label is not None and slice_label is not None:
        return np.array(X_data), np.array(x_filename), np.array(disease_label), np.array(slice_label)
    return np.array(X_data), np.array(x_filename)

def load_png_slice(X_data_path):
    """
    :param X_data_path:
    :return: (S, H, W), S:number of slices (P * 36), H:height, W:width
    """

    X_data = [] # result as a list of slice samples
    x_filename = []

    for path in X_data_path:
        pixels = misc.imread(path)
        X_data.append(pixels)

        pid_filename = os.path.basename(path) # remove .nii
        x_filename.append(pid_filename)

    return np.array(X_data), np.array(x_filename)

# revise lines for modeling slices labeled by clinical diagnosis
def load_nii_slice_data(X_data_path, y_label, mask_path=None):
    """

    :param X_data_path:
    :param y_label:
    :param mask_path:
    :return: (S, H, W), S:number of slices (P * 36), H:height, W:width
    """

    slice_label_ref = "/root/PycharmProjects/1903_bapl_gda/Slice grading_final.xlsx"
    label_ref_df_bapl2 = pd.read_excel(slice_label_ref, sheetname="BAPL2")

    # mask_path = "/root/PycharmProjects/18_experiment_conference_idea/nifti_mask/brainmask_grey_resize_79_95_68.nii"
    if mask_path is not None:
        mask_nib = nib_load(mask_path)
        mask_mat = np.array(mask_nib.get_data())
        mask_mat = np.rot90(mask_mat, k=1, axes=(0, 1))

    X_data = [] # result as a list of slice samples
    y_label_new = []
    x_filename = []

    for path, label in zip(X_data_path, y_label):

        whole_voxel = nib_load(path)
        whole_voxel = np.array(whole_voxel.get_data())
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(0, 1))

        if mask_path is not None:
            whole_voxel = whole_voxel*mask_mat[:,:,:,0]

        whole_voxel = np.array(whole_voxel[:, :, 14:50]).astype(np.float32)
        whole_voxel = np.transpose(whole_voxel, [2, 0, 1])
        X_data.append(whole_voxel)

        pid_filename = os.path.basename(path)[:-4]
        #print("[!] debugging, path, label", path, label)
        if label == 1:
            pid_label = np.array(label_ref_df_bapl2[pid_filename][14:50].tolist())-1
            #print("[!] debugging, pid_label", pid_label)
        else:
            pid_label = np.array([label]*len(whole_voxel))
        y_label_new.append(pid_label)
        filename = os.path.basename(path)
        x_filename.append([filename + "_" + str(ind) for ind in range(14, 50)])

    X_data = np.concatenate(X_data)
    y_label_new = np.concatenate(y_label_new)
    x_filename = np.concatenate(x_filename)
    return np.array(X_data), np.array(y_label_new), np.array(x_filename)


def load_nii_15_50_slice_data(X_data_path, y_label, mask_path=None):
    """

    :param X_data_path:
    :param y_label:
    :param mask_path:
    :return: (P, S, H, W), P:number of patients, S:number of slices, H:height, W:width
    """
    slice_label_ref = "/root/PycharmProjects/1903_bapl_gda/Slice grading_final.xlsx"
    label_ref_df_bapl2 = pd.read_excel(slice_label_ref, sheetname="BAPL2")

    # mask_path = "/root/PycharmProjects/18_experiment_conference_idea/nifti_mask/brainmask_grey_resize_79_95_68.nii"
    if mask_path is not None:
        mask_nib = nib_load(mask_path)
        mask_mat = np.array(mask_nib.get_data())
        mask_mat = np.rot90(mask_mat, k=1, axes=(0, 1))

    X_data = [] # result as a list of slice samples
    y_label_new = []
    x_filename = []
    for path, label in zip(X_data_path, y_label):

        whole_voxel = nib_load(path)
        whole_voxel = np.array(whole_voxel.get_data())
        whole_voxel = np.rot90(whole_voxel, k=1, axes=(0, 1))

        if mask_path is not None:
            whole_voxel = whole_voxel*mask_mat[:,:,:,0]

        whole_voxel = np.array(whole_voxel[:, :, 14:50]).astype(np.float32)
        whole_voxel = np.transpose(whole_voxel, [2, 0, 1])
        X_data.append(whole_voxel)

        pid_filename = os.path.basename(path)[:-4]
        if label == 1:
            pid_label = np.array(label_ref_df_bapl2[pid_filename][14:50].tolist())-1
            #print("[!] debugging, pid_label", pid_label)
        else:
            pid_label = np.array([label] * len(whole_voxel))
        y_label_new.append(pid_label)
        filename = os.path.basename(path)
        x_filename.append([filename + "_" + str(ind) for ind in range(14, 50)])

    return np.array(X_data), np.array(y_label_new), np.array(x_filename)


def decide_p_based_bapl(prediction):
    """
    :param prediction: prediction for a patient's slices
    :return: decision for bapl score which he have
    """
    if 2 in prediction:
        return 2
    elif 1 in prediction:
        return 1
    else :
        return 0

def decide_p_based_clinical_nc_ad(prediction):
    """
    :param prediction: prediction for a patient's slices
    :return: decision for bapl score which he have
    """
    if 0 in prediction:
        return 0
    else:
        return 1

# RULE 1 : decide the largest classes as a final decision for patient
def decide_p_based_clinical_nc_mci_ad(prediction):
    """
    :param prediction: prediction for a patient's slices
    :return: decision for bapl score which he have
    """
    classes = np.unique(prediction)
    if 0 in prediction:
        return 0
    elif 1 in prediction:
        return 1
    else :
        return 2


def save_pred_label(label, pred, save_filepath, onehot_label=True, filename_list = None):
    """
    :param label:
    :param pred:
    :param save_filepath:
    :param onehot_label:
    :param filename_list: id indicating for each instance infered from trained model_name, which is helpful to trace data for debugging
    :return: None, this function just store excel data arranging table with label and prediction of trained model_name
    """
    filename_list = np.array(filename_list).tolist()
    if not onehot_label : # when onehot_label is False, the shape is (batch_size, ), i.e, expressing the data with 1 column
        num_classes = len(np.unique(label))
        log_dict = dict()
        log_dict["label"] = np.array(label).tolist()
        log_dict["pred"] = np.array(pred).tolist()
        categorical_label = to_categorical(label, num_classes)
        for ind in range(num_classes):
            log_dict["pred_"+str(ind)] = np.array(categorical_label)[:,ind].tolist()
        if filename_list:
            log_dict["pid"] = filename_list
    else: # when onehot_label is True, the shape of pred AND label is (batch_size, num_classes), i.e expressing the data with many column
        num_classes = len(label[0])

        pred_ = np.array(pred).argmax(axis=1)
        label_ = np.array(label).argmax(axis=1)
        print("pred_", pred_.shape)

        log_dict = dict()
        log_dict["label"] = np.array(label_).tolist()
        log_dict["pred"] = np.array(pred_).tolist()
        for ind in range(num_classes):
            log_dict["pred_"+str(ind)] = np.array(pred)[:,ind].tolist()

        if filename_list:
            log_dict["pid"] = filename_list


    print("label", len(log_dict["label"]))
    print("pred", len(log_dict["pred"]))
    print("pid", len(log_dict["pid"]))

    data = pd.DataFrame(log_dict)
    data.to_excel(save_filepath)

# so return list of element for smallest element from input
def find_smallest_figures(input_list):
    res = []
    minval = 2000000000
    for ind, elem in enumerate(input_list):
        if elem < minval:
            res = []
            minval = elem
            res.append(ind)
        elif elem == minval:
            res.append(ind)
    return res, minval

def to_categorical_concatenate_list(label_list):
    label_list = [to_categorical(y) for y in label_list]
    res = np.concatenate(label_list, axis=1).astype(np.int32)
    return res


def make_all_two_cond_label_case(label_num_on_cond_list, iter_second=True):
    """
    # len(label_num_on_cond_list)== number of conditions

    :param label_num_on_cond_list: 
    :return: (len(label_num_on_cond_list[0])*len(label_num_on_cond_list[1]), len(label_num_on_cond_list[0])+len(label_num_on_cond_list[1]))
    """
    sampled_classes = np.arange(0, label_num_on_cond_list[0])  # (36, )
    sampled_labels = np.arange(0, label_num_on_cond_list[1])  # (2, )
    sampled_classes_list = []
    sampled_labels_list = []
    if iter_second is True:
        for label_ in sampled_labels:
            for class_ in sampled_classes:
                sampled_classes_list.append(class_)
                sampled_labels_list.append(label_)
    else:
        for class_ in sampled_classes:
            for label_ in sampled_labels:
                sampled_classes_list.append(class_)
                sampled_labels_list.append(label_)

    sampled_classes_labels = to_categorical_concatenate_list((sampled_classes_list, sampled_labels_list))  # (1, 39)
    return sampled_classes_labels

def random_sampling(data_list, numSample, redundancy, option="uniform"):
    if len(data_list) < numSample:
        print("len(data_list) has to be bigger than numSample")
        return
    data_list = np.array(data_list)
    if option == "uniform":
        ind = np.random.choice(len(data_list), numSample, replace=redundancy)
    return data_list[ind].tolist()

if __name__ == '__main__':
    test = [2, 3, -1, 6, -1, 1, 1, 4, 1]
    print(find_smallest_figures(test))

