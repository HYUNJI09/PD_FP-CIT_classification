"""
https://github.com/Dong-A-NM/Utilities

seungyeoniii 2020-07-13 commit
56668cf

revised : 210505

"""

import os
import sys
import io
import base64

import tqdm
import math
import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk
import matplotlib.pyplot as plt

import pydicom
from nibabel import load as nib_load
from custom_exception import *

#
class ImageDataIO():
    # arrange dimension for each type of view (axial / saggital / coronal)
    def __init__(self, extention, dataDim, instanceIsOneFile, modeling="2D", view=None, mask_path=None):
        """
        :param extention: extention of file to read
        :param dataDim: data dim to treat, '2D' of image or '3D' of image, in a file; 
        :param instanceIsOneFile: how is an instance stored for one instance; True'one_file' / 'multiple_files' 
        :param modeling: how to handle an instance, e.g 2D or 3D
        :param view: 
        """

        # if is2D == True and view is not None:
        #     print("is2D variable can't have view property!")
        #     return
        self._extention = extention
        self._dataDimInOneFile = dataDim
        self._instanceIsOneFile = instanceIsOneFile
        self._modeling = modeling
        self._view = view
        self._mask_path = mask_path

        return

    # handling one file
    def read_file(self, source_path):
        if self._extention == "jpg" or self._extention == "jpeg" or self._extention == "png":
            return self._read_popular_img(source_path)
        elif self._extention == "dcm" or self._extention == "dicom":
            return self._read_dicom_img(source_path)
        elif self._extention == "nii" or self._extention == "nifti":
            return self._read_nifti_img(source_path)

    # reading jpeg or png
    def _read_popular_img(self, source_path):
        """
        if one file's type to read is either jpg or png, we don't have to consider the case the file have 3D data.
        :param source_path: input path for one instance to read from jpg or png file
        :return: a image or images consisting of one instance
        """
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            img = Image.open(source_path)
            return img

        elif self._dataDimInOneFile == "2D" and not self._instanceIsOneFile:
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            imgs = []
            for child_path in child_paths:
                img = Image.open(child_path)
                imgs.append(img)

            return imgs
        elif self._dataDimInOneFile == "3D":
            raise CustomException("when input file's extention is jpg/png, the case the input is 3D is not considered")

    # reading dicom
    def _read_dicom_img(self, source_path):
        """
        :param source_path: input path for one instance to read from dicom file
        :return: a image or images consisting of one instance
        """
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            dcm_img = np.array(pydicom.read_file(source_path).pixel_array)
            return Image.fromarray(dcm_img)
        elif self._dataDimInOneFile == "2D" and not self._instanceIsOneFile:  # source_path is list of dcm files for one subject
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            dcm_imgs = []
            for child_path in child_paths:
                dcm_img = pydicom.read_file(child_path)  # dcmread
                dcm_imgs.append(dcm_img)
            # sorting and save
            dcm_imgs.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            # dcm_imgs = np.array([dcm_img.pixel_array for dcm_img in dcm_imgs])
            dcm_imgs_list = []
            for ind, dcm_img in enumerate(dcm_imgs):
                try:
                    dcm_imgs_list.append(dcm_img.pixel_array)
                    # print("debug, dcm_img.pixel_array", type(dcm_img.pixel_array))
                    # print("debug, dcm_img.pixel_array", dcm_img.pixel_array.dtype)
                except NotImplementedError as nie:  # when data is stored as byte array
                    print(nie)
                    tmp_bytes = dcm_img.PixelData

                    # print("debug, tmp_bytes", type(tmp_bytes))
                    # print("debug, pixel_array", dcm_img.pixel_array)
                    # 1
                    # im = Image.frombytes("LA", (400, 400), tmp_bytes)
                    # im.save(os.path.join("./fish", str(ind)+"_test.jpg"), 'JPG')

                    # 2
                    with open(os.path.join("./fish", str(ind) + "_test.jpg"), 'wb') as f:
                        # f.write(tmp_bytes.decode("latin1"))
                        # tmp = base64.b64decode(tmp_bytes)

                        f.write(str(tmp_bytes))

                    # tmp_mat = np.frombuffer(tmp_bytes, dtype=np.uint8)

                    # 3

                    nparr = np.fromstring(tmp_bytes, np.uint16)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                    # rsize = dcm_img.Rows
                    # csize = dcm_img.Columns
                    # tmp_mat.reshape(rsize, csize)

                    dcm_imgs_list.append(img_np)

                    ##
                    # image = Image.open(io.BytesIO(tmp_bytes))
                    pass

            # print("[!!] debug len(dcm_imgs_list)", len(dcm_imgs_list))
            dcm_imgs_list = np.array(dcm_imgs_list)
            # print("debug type of dcm_imgs_list", dcm_imgs_list.dtype)
            # print("[!!] debug len(dcm_imgs_list)", dcm_imgs_list.shape)

            # checking image
            # np_data = self.convert_PIL_to_numpy(dcm_imgs_list)
            # self.show_one_img(dcm_imgs_list, is2D=False, cmap=None)

            if self._view == "axial" or self._view == "transaxial":
                # dcm_imgs = dcm_imgs
                dcm_imgs = dcm_imgs_list
            elif self._view == "coronal":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(1, 2))  #
                dcm_imgs = np.rot90(dcm_imgs, k=3, axes=(0, 1))
            elif self._view == "saggital":
                dcm_imgs = np.rot90(dcm_imgs, k=1, axes=(0, 2))  #

                ##########################3
                # if nib_img.min() < 0:
                #     print(
                #         "[!] Warning : Input image argument has a negative pixel or voxel. This program make it to be on [0, 255]")
                #     nib_img = (nib_img - nib_img.min()) / (nib_img.max() - nib_img.min()) * 255
                # elif nib_img.max() > 255.0:  # raw img was not encoded in uint8
                #     nib_img = (nib_img / 65535) * 255

                # print("3D shape", nib_img.shape)
                #return [Image.fromarray(img) for img in dcm_imgs]

            else:
                raise CustomException(
                    "the state that _instanceIsOneFile is False is not defined yet in Nifti type")
            #############################
            print("[!] debug, dcm_imgs", dcm_imgs.min(), dcm_imgs.max(), type(dcm_imgs))
            dcm_imgs = self._change_image_intensity_range(dcm_imgs)
            return [Image.fromarray(dcm_img_pixels) for dcm_img_pixels in dcm_imgs]
        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            dcm_img = np.array(pydicom.read_file(source_path).pixel_array)
            dcm_img = self._change_image_intensity_range(dcm_img)
            return Image.fromarray(dcm_img)
        else:
            raise CustomException(
                "the state that _dataDimInOneFile is 3D and _instanceIsOneFile is False is not defined")

    # reading nifti
    def _read_nifti_img(self, source_path):
        """
        :param source_path: input path for one instance to read from nitfi file
        :return: a image or images consisting of one instance
        """
        """
        :param source_path: 
        :return: list of Image obj 
        """
        if self._mask_path is not None:
            mask_mat = nib_load(self._mask_path)
            mask_mat = np.array(mask_mat.get_data())  # (W, H, D)
			
        nib_img = nib_load(source_path)
        nib_img = np.array(nib_img.get_data()) # [debug] ############## 19602
        # print("[debug] ##############", len(np.unique(nib_img)))
        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            nib_img = np.rot90(nib_img, k=1, axes=(0, 1))
            return Image.fromarray(nib_img)
        elif self._dataDimInOneFile == "3D" and not self._instanceIsOneFile:
            child_paths = [os.path.join(source_path, path) for path in os.listdir(source_path)]
            nii_imgs = []
            for child_path in child_paths:
                nii_img = nib_load(child_path)
                nii_imgs.append(nii_img)
            nii_imgs = np.array([nii_img.get_data() for nii_img in nii_imgs])
            if self._view == "axial" or self._view == "transaxial":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 1))  # (95, 79, 68)
            elif self._view == "saggital":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(0, 2))  #
            elif self._view == "coronal":
                nii_imgs = np.rot90(nii_imgs, k=1, axes=(1, 2))  #
                nii_imgs = np.rot90(nii_imgs, k=3, axes=(0, 1))

            #nii_imgs = nii_imgs.astype(np.uint8)
            nii_imgs = np.transpose(nii_imgs, [2, 0, 1])
            return [Image.fromarray(nii_img) for nii_img in nii_imgs]
        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            if self._view == "axial" or self._view == "transaxial":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 1))  # (95, 79, 68)
                if self._mask_path is not None:
                    mask_mat = np.rot90(mask_mat, k=1, axes=(0, 1))  # (H, W, D)
                    mask_mat = np.array(mask_mat[7:102, 6:85, 11:79]).astype(np.float32)
                    nib_img = nib_img * mask_mat[:, :, :]
            elif self._view == "saggital":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 2))  #
            elif self._view == "coronal":
                nib_img = np.rot90(nib_img, k=1, axes=(1, 2))  #
                nib_img = np.rot90(nib_img, k=3, axes=(0, 1))

            # if nib_img.min() < 0:
            #     print(
            #         "[!] Warning : Input image argument has a negative pixel or voxel. This program make it to be on [0, 255]")
            #     nib_img = (nib_img - nib_img.min()) / (nib_img.max() - nib_img.min()) * 255
            # elif nib_img.max() > 255.0:  # raw img was not encoded in uint8
            #     nib_img = (nib_img / 65535) * 255
            # print("[debug] ##############", len(np.unique(nib_img)))

            # nib_img = self._change_image_intensity_range(nib_img)
            # nib_img = nib_img.astype(np.uint8)
            nib_img = np.transpose(nib_img, [2, 0, 1])
            # print("3D shape", nib_img.shape)
            #return [Image.fromarray(img) for img in nib_img]
            return nib_img

        else:
            raise CustomException(
                "the state that _instanceIsOneFile is False is not defined yet in Nifti type")




    @classmethod
    def pad_2d_image_by_size(cls, input_img, target_img_size, single_obj=True, dimData="2D"):
        """

        :param input_img:
        :param target_img_size: d
        :param single_obj:
        :param dimData:
        :return:
        """
        input_img = np.array(input_img)
        if input_img.min()<0:
            raise CustomException("[!] input_img doens't be allowed to have a negative.")
        pad_board = None
        if single_obj and dimData == "2D":
            if len(input_img.shape) ==3:
                h, w, c = input_img.shape
                pad_board = np.zeros(target_img_size)
                pad_board -= 1
            elif len(input_img.shape) == 2:
                h, w = input_img.shape

            board_h = target_img_size[0]
            board_w = target_img_size[1]
            board_h_ind = int((board_h - h)/2)
            board_w_ind = int((board_w - w)/2)
            pad_board[board_h_ind:board_h_ind+h, board_w_ind:board_w_ind+w] = input_img
            pad_board[pad_board<0] = 0
            return np.array(pad_board)
        elif single_obj and dimData == "3D":  # (D, H, W, 1)
            pad_board = np.zeros(target_img_size)
            pad_board -= 1

            if len(input_img.shape) ==4:
                d, h, w, c = input_img.shape
            elif len(input_img.shape) == 3:
                d, h, w = input_img.shape

            # board_d = target_img_size[0]
            board_h = target_img_size[1]
            board_w = target_img_size[2]
            board_h_ind = int((board_h - h) / 2)
            board_w_ind = int((board_w - w) / 2)
            for _d_ind in range(d):
                pad_board[_d_ind][board_h_ind:board_h_ind+h, board_w_ind:board_w_ind+w] = input_img[_d_ind]
            pad_board[pad_board<0] = 0

            return np.array(pad_board)
        elif not single_obj and dimData == "2D":  # (N, H, W, C)
            if len(input_img.shape) == 4: #
                n, h, w, c = input_img.shape
                pad_board = np.zeros(target_img_size)
                pad_board -= 1
            elif len(input_img.shape) == 3:
                h, w = input_img.shape

            board_h = target_img_size[0]
            board_w = target_img_size[1]
            board_h_ind = int((board_h - h) / 2)
            board_w_ind = int((board_w - w) / 2)
            pad_board[board_h_ind:board_h_ind + h, board_w_ind:board_w_ind + w] = input_img
            pad_board[pad_board < 0] = 0
            return np.array(pad_board)
        elif not single_obj and dimData == "3D":  # (N, D, H, W, C)
            processed_3d_img_list = []
            for _3d_img in input_img:
                processed_img_list = []
                for _img in _3d_img:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
                    img_c = clahe.apply(_img)
                    processed_img_list.append(img_c)
                processed_3d_img_list.append(processed_img_list)
            return np.array(processed_3d_img_list)
        return

    def _change_image_intensity_range(self, instance_img, encoding_to_change=8):
        instance_img = np.array(instance_img)
        _min = instance_img.min().astype(np.int64)
        _max = instance_img.max().astype(np.int64)
        _resolution_range = _max - _min

        _encoding = None

        if _resolution_range > 255:
            _encoding = 16
        else :
            _encoding = 8

        instance_img = (instance_img - _min) / 2 ** _encoding
        if encoding_to_change == 8:
            return (instance_img*(2**encoding_to_change)).astype(np.uint8)
        elif encoding_to_change == 16:
            return (instance_img * (2 ** encoding_to_change)).astype(np.uint16)

    def _resizing_channel(self, img_obj, resize_size, channel_size=None, dataDim=None):
        """
        :param img_obj: img_obj have to be a list of images whether self._modeling is "2D" or "3D"
        :param resize_size:
        :param channel_size:
        :return:
        """
        if dataDim is not None:
            self._modeling = dataDim
        if sys.getsizeof(img_obj) == 0:
            print("size of img_data object is 0")
            return None
        # if isinstance(img_obj, list):
        #     img_shape = img_obj[0].size
        # else :
        #     img_shape = img_obj.size

        # resize
        if resize_size is not None:
            if self._modeling == "3D":  # (N, D, H, W, C)
                sub_list = []
                for sub in img_obj:
                    imgs = [img.resize(resize_size) for img in sub]
                    sub_list.append(imgs)
                img_obj = sub_list

            elif self._modeling == "2D":  # (N, H, W, C)
                # img_data = np.array([cv2.resize(np.array(x), resize_size) for x in img_data]).astype(np.float32)
                if isinstance(img_obj, list):
                    img_obj = [img.resize(resize_size) for img in img_obj]
                else:
                    img_obj = img_obj.resize(resize_size)

        # check channel
        if self._modeling == "3D":
            sub_list = []
            for sub in img_obj:
                if channel_size is not None and channel_size == 1:
                    # imgs = []
                    imgs = [img.convert("L") for img in sub]
                elif channel_size is not None and channel_size == 3:
                    imgs = [img.convert("RGB") for img in sub]
                elif channel_size is not None and channel_size == 4:
                    imgs = [img.convert("RGBA") for img in sub]
                sub_list.append(imgs)
            img_obj = sub_list
        elif self._modeling == "2D":  # isinstance(img_obj, list):
            if channel_size is not None and channel_size == 1:
                img_obj = [img.convert("L") for img in img_obj]
            elif channel_size is not None and channel_size == 3:
                img_obj = [img.convert("RGB") for img in img_obj]
            elif channel_size is not None and channel_size == 4:
                img_obj = [img.convert("RGBA") for img in img_obj]
        else:  # img_obj is just one file
            if channel_size is not None and channel_size == 1:
                img_obj = img_obj.convert("L")
            elif channel_size is not None and channel_size == 3:
                img_obj = img_obj.convert("RGB")
            elif channel_size is not None and channel_size == 4:
                img_obj = img_obj.convert("RGBA")

        return img_obj

    def read_files_from_dir(self, source_dir):
        """
        :param source_dir:  input path for multiple instance to read from nitfi file
        :return:
        """
        child_full_paths = [os.path.join(source_dir, path) for path in os.listdir(source_dir)]
        file_list = []
        filename_list = []
        for child_full_path in child_full_paths:
            file_list.append(self.read_file(child_full_path))
            filename_list.append(os.path.basename(child_full_path))
        return file_list, filename_list  # list of PIL Image obj

    def convert_PIL_to_numpy(self, img_obj, dataDim=None):

        if dataDim is not None:
            self._dataDimInOneFile = dataDim
        if isinstance(img_obj, list):
            if self._dataDimInOneFile == "3D":  # input is 4d (list of 3d)
                sub_list = []
                for sub in img_obj:
                    img_list = []
                    for img in sub:
                        # print(np.asarray(img))
                        img_list.append(np.asarray(img))
                    img_list = np.array(img_list)
                    sub_list.append(img_list)
                return np.array(sub_list)

            elif self._dataDimInOneFile == "2D" and self._instanceIsOneFile is False:  # input is 4d (list of 3d)
                sub_list = []
                for sub in img_obj:
                    img_list = []
                    for img in sub:
                        # print(np.asarray(img))
                        img = np.asarray(img)
                        # img = self._clahe_img(img)
                        img_list.append(np.asarray(img))
                    img_list = np.array(img_list)
                    sub_list.append(img_list)
                return np.array(sub_list)
            elif self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
                return np.array([np.array(img) for img in img_obj])

        else:  # img_obj is just one instance not list
            return np.asarray(img_obj)

    def convert_numpy_to_PIL(self, img_obj, single_obj=False, dataDim=None):
        if isinstance(img_obj, np.ndarray) and img_obj.dtype == np.float64 and img_obj.flags.contiguous:
            img_obj = self._change_image_intensity_range(img_obj)
            img_obj = img_obj.astype(np.uint8)

        if dataDim is not None:
            self._dataDimInOneFile = dataDim
        if single_obj:
            return Image.fromarray(img_obj)
        # print("convert_numpy_to_PIL",img_obj)
        if self._dataDimInOneFile == "3D":
            sub_list = []
            for sub in img_obj:
                img_list = []
                for img in sub:
                    # print(np.asarray(img))
                    img_list.append(Image.fromarray(img))
                sub_list.append(img_list)
            return sub_list

        elif self._dataDimInOneFile == "2D":
            return [Image.fromarray(img) for img in img_obj]

    def convert_PIL_to_ImageTk(self, pil_img_list, single_obj=False):
        if single_obj:
            return ImageTk.PhotoImage(pil_img_list)
        if self._dataDimInOneFile == "2D":
            return [ImageTk.PhotoImage(pil_img) for pil_img in pil_img_list]
        elif self._dataDimInOneFile == "3D":
            sub_list = []
            for sub in pil_img_list:
                img_list = []
                for img in sub:
                    # print(np.asarray(img))
                    img_list.append(ImageTk.PhotoImage(img))
                img_list = img_list
                sub_list.append(img_list)
            return sub_list

    # convert PIL object into ImageTk obj, for one sample
    # def convert_PIL_to_ImageTk_one_sample(self, pil_img):
    #     if single_obj:
    #         return ImageTk.PhotoImage(pil_img_list)
    #     if self._dataDim == "2D":
    #         return [ImageTk.PhotoImage(pil_img) for pil_img in pil_img_list]
    #     elif self._dataDim == "3D":
    #         sub_list = []
    #         for sub in pil_img_list:
    #             img_list = []
    #             for img in sub:
    #                 # print(np.asarray(img))
    #                 img_list.append(ImageTk.PhotoImage(img))
    #             img_list = img_list
    #             sub_list.append(img_list)
    #         return sub_list

    @classmethod
    def show_one_img(cls, imgs, is2D=True, cmap=None):
        if is2D:
            plt.imshow(imgs, cmap)
            plt.show()
        else:
            num_imgs = len(imgs)
            board_size = math.ceil(math.sqrt(num_imgs))
            r = c = board_size
            img_ind = 0
            for r_ind in range(r):
                for c_ind in range(c):
                    if img_ind >= len(imgs):
                        plt.show()
                        return
                    c_ind_ = c_ind + 1
                    plt.subplot(r, c, r_ind * c + c_ind_)
                    plt.imshow(imgs[img_ind], cmap=cmap)
                    img_ind = img_ind + 1
            plt.show()
        return

    @classmethod
    def show_one_img_v2(cls, imgs, is2D=True, cmap=None):
        if is2D:
            plt.imshow(imgs, cmap)
            plt.show()
        else:
            num_imgs = len(imgs)
            board_size = math.ceil(math.sqrt(num_imgs))
            r = c = board_size
            img_ind = 0
            for r_ind in range(r):
                for c_ind in range(c):
                    if img_ind >= len(imgs):
                        plt.show()
                        return
                    c_ind_ = c_ind + 1
                    # plt.subplot(r, c, r_ind* c +c_ind_)
                    plt.imshow(imgs[img_ind], cmap=cmap)
                    img_ind = img_ind + 1
            plt.show()
        return


    @classmethod
    def show_one_img_v3(self, img, is2D=True, cmap=None):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D : # one 2D image
            plt.imshow(img, cmap)
            plt.show()
        else : # one 3D image
            len_depth = len(img)
            num_rows = img.shape[1]
            num_cols = img.shape[2]
            grid_size = math.ceil(math.sqrt(len_depth))
            r_board_size = grid_size*img.shape[1]
            c_board_size = grid_size * img.shape[2]
            grid_board = np.zeros([r_board_size, c_board_size])
            img_ind = 0
            for r_grid_ind in range(grid_size):
                for c_grid_ind in range(grid_size):
                    grid_board_c_start_ind = c_grid_ind*num_cols
                    grid_board_r_start_ind = r_grid_ind*num_rows
                    grid_board[grid_board_r_start_ind:grid_board_r_start_ind+num_rows, grid_board_c_start_ind:grid_board_c_start_ind+num_cols] = img[img_ind, :, :]

                    img_ind = img_ind+1
                    if img_ind >= len(img):
                        plt.imshow(grid_board, cmap=cmap)
                        plt.colorbar()
                        plt.show()

                        return

        return

    @classmethod
    def get_one_img_v3(self, img, is2D=True, cmap=None):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D:  # one 2D image
            return img
        else:  # one 3D image
            len_depth = len(img)
            num_rows = img.shape[1]
            num_cols = img.shape[2]
            grid_size = math.ceil(math.sqrt(len_depth))
            r_board_size = grid_size * img.shape[1]
            c_board_size = grid_size * img.shape[2]
            grid_board = np.zeros([r_board_size, c_board_size])
            img_ind = 0
            for r_grid_ind in range(grid_size):
                for c_grid_ind in range(grid_size):
                    grid_board_c_start_ind = c_grid_ind * num_cols
                    grid_board_r_start_ind = r_grid_ind * num_rows
                    grid_board[grid_board_r_start_ind:grid_board_r_start_ind + num_rows,
                    grid_board_c_start_ind:grid_board_c_start_ind + num_cols] = img[img_ind, :, :]

                    img_ind = img_ind + 1
                    if img_ind >= len(img):
                        return grid_board

        return

    def holdout_np_ver(self, img_obj, train_rate, cluster_bootstrap=False, cluster_func=None):
        num_img_to_holdout = len(img_obj)
        num_train = int(num_img_to_holdout * train_rate)
        num_test = num_img_to_holdout - num_train

        if cluster_bootstrap:
            if self._instanceIsOneFile:  # shape of img_obj, (N, H, W, C)

                return
            else:  # shape of img_obj, (B, D, H, W, C)
                return
            return
        else:
            # shape of img_obj, (N, H, W, C)

            return

    def _get_unchoiced_ind(self, choiced_ind_list, num_total):
        unchoiced_ind_list = []
        for ind in range(len(num_total)):
            if ind not in choiced_ind_list:
                unchoiced_ind_list.append(ind)
        return np.array(unchoiced_ind_list)

    def convertUint8(self, img_obj):
        """
        1. dicom :16bit -> the very info?? 0~65535 -> 16? 8?
        [0,1] / [-1,1]

        2. convertion into uint8


        :param img_obj:
        :param _:
        :return:
        """
        sub_list = []
        for sub in img_obj:
            img_list = []
            for img in sub:
                pix = np.asarray(img)
                pix_norm = (pix - pix.min()) / (pix.max() - pix.min())
                # pix_norm = (pix - pix.min()) / (65535 - pix.min())
                pix_norm = pix_norm * 255
                img_list.append(Image.fromarray(pix_norm))
            sub_list.append(img_list)

        return sub_list

    def _cropping(self, pil_img_list, bbox):

        if self._dataDimInOneFile == "2D" and self._instanceIsOneFile:
            return [pil_img.crop(bbox) for pil_img in pil_img_list]
        elif (self._dataDimInOneFile == "2D" and self._instanceIsOneFile is False) or \
                (self._dataDimInOneFile == "3D" and self._instanceIsOneFile):  # 4d
            sub_list = []
            for sub in pil_img_list:
                img_list = []
                for img in sub:
                    # print(np.asarray(img))
                    img_list.append(img.crop(bbox))

                sub_list.append(img_list)
            return sub_list

    def histogram_img(self, pil_img_list, save_path):

        img_list = []
        for img in pil_img_list:
            pix = np.asarray(img)
            img_list.append(pix.flatten())  # [1d,1d,1d...]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
        return

    def histogram_img_old(self, pil_img_list, filename, hist_path):

        img_list = []

        for img in pil_img_list:
            print("debug histogram", img)

            # pix = np.asarray(img, dtype='uint8')
            pix = np.array(img)
            # range_ = len(np.unique(pix))
            # plt.hist(pix, range_)
            # plt.show()
            img_list.append(pix.flatten())

        img_list = np.array(img_list)
        # imgs_flat = img_list.flatten()

        for ind, img in enumerate(img_list):
            # print("bins", bins)
            # print("bins min max", bins.min(), bins.max())
            plt.hist(img, 256)
            # plt.xlim(0, 255)
            plt.ylim(0, 20000)
            save_path = os.path.join(hist_path, str(filename[ind]) + '.jpg')
            plt.savefig(save_path)
            # plt.show()
            plt.close()
            print("debug save_path", save_path)
        print("[!]histogram completed!")
        return

    @classmethod
    def clahe_img(self, input_img, single_obj=True, dimData="2D"):
        if single_obj and dimData == "2D":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            img_c = clahe.apply(input_img)
            return np.array(img_c)
        elif single_obj and dimData == "3D":  # (D, H, W, 1)
            processed_img_list = []
            for _img in input_img:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
                img_c = clahe.apply(_img)
                processed_img_list.append(img_c)
            return np.array(processed_img_list)
        elif not single_obj and dimData == "2D":  # (N, H, W, C)
            processed_img_list = []
            for _img in input_img:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
                img_c = clahe.apply(_img)
                processed_img_list.append(img_c)
            return np.array(processed_img_list)
        elif not single_obj and dimData == "3D":  # (N, D, H, W, C)
            processed_3d_img_list = []
            for _3d_img in input_img:
                processed_img_list = []
                for _img in _3d_img:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
                    img_c = clahe.apply(_img)
                    processed_img_list.append(img_c)
                processed_3d_img_list.append(processed_img_list)
            return np.array(processed_3d_img_list)

    @classmethod
    def fourier_transform(cls, input_img, single_obj=True, dimData="2D"):
        if single_obj and dimData == "2D":
            f = np.fft.fft2(input_img)  # 이미지에 푸리에 변환 적용
            fshift = np.fft.fftshift(f)  # 분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴. 중앙에 저주파가 모이게 됨.
            magnitude_spectrum = 20 * np.log(np.abs(fshift))  # spectrum 구하는 수학식.

            rows, cols = input_img.shape
            crow, ccol = rows / 2, cols / 2  # 이미지의 중심 좌표

            # 중앙에서 10X10 사이즈의 사각형의 값을 1로 설정함. 중앙의 저주파를 모두 제거
            # 저주파를 제거하였기 때문에 배경이 사라지고 경계선만 남게 됨.
            d = 10
            fshift[crow - d:crow + d, ccol - d:ccol + d] = 1

            # 푸리에 변환결과를 다시 이미지로 변환
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            return np.array(img_back)
        elif single_obj and dimData == "3D":  # (D, H, W, 1)
            processed_img_list = []
            for _img in input_img:
                f = np.fft.fft2(_img[:,:,0])  # 이미지에 푸리에 변환 적용
                fshift = np.fft.fftshift(f)  # 분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴. 중앙에 저주파가 모이게 됨.
                magnitude_spectrum = 20 * np.log(np.abs(fshift))  # spectrum 구하는 수학식.

                rows, cols = _img[:,:,0].shape
                crow, ccol = rows / 2, cols / 2  # 이미지의 중심 좌표

                # 중앙에서 10X10 사이즈의 사각형의 값을 1로 설정함. 중앙의 저주파를 모두 제거
                # 저주파를 제거하였기 때문에 배경이 사라지고 경계선만 남게 됨.
                d = 10
                fshift[crow - d:crow + d, ccol - d:ccol + d] = 1

                # 푸리에 변환결과를 다시 이미지로 변환
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                processed_img_list.append(img_back)
            return np.array(processed_img_list)
        elif not single_obj and dimData == "2D":  # (N, H, W, C)
            processed_img_list = []
            for _img in input_img:
                f = np.fft.fft2(_img)  # 이미지에 푸리에 변환 적용
                fshift = np.fft.fftshift(f)  # 분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴. 중앙에 저주파가 모이게 됨.
                magnitude_spectrum = 20 * np.log(np.abs(fshift))  # spectrum 구하는 수학식.

                rows, cols = _img.shape
                crow, ccol = rows / 2, cols / 2  # 이미지의 중심 좌표

                d = 10
                fshift[crow - d:crow + d, ccol - d:ccol + d] = 1

                # 푸리에 변환결과를 다시 이미지로 변환
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                processed_img_list.append(img_back)
            return np.array(processed_img_list)
        elif not single_obj and dimData == "3D":  # (N, D, H, W, C)
            processed_3d_img_list = []
            for _3d_img in input_img:
                processed_img_list = []
                for _img in _3d_img:
                    f = np.fft.fft2(_img[:,:,0])  # 이미지에 푸리에 변환 적용
                    fshift = np.fft.fftshift(f)  # 분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴. 중앙에 저주파가 모이게 됨.
                    magnitude_spectrum = 20 * np.log(np.abs(fshift))  # spectrum 구하는 수학식.

                    rows, cols = _img[:,:,0].shape
                    crow, ccol = rows / 2, cols / 2  # 이미지의 중심 좌표

                    d = 10
                    fshift[int(crow - d):int(crow + d), int(ccol - d):int(ccol + d)] = 1

                    # 푸리에 변환결과를 다시 이미지로 변환
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = np.fft.ifft2(f_ishift)
                    img_back = np.abs(img_back)
                    processed_img_list.append(img_back[:,:,None])
                processed_3d_img_list.append(processed_img_list)
            return np.array(processed_3d_img_list)

if __name__ == "__main__":
    print("hello world")

    # extention = "png"
    # is2D = False
    # view="axial"
    # source_path = "C:\\Users\\NM\\PycharmProjects\\Med_Labeling\\images.png"
    # idio = ImageDataIO(extention, is2D)
    # img = idio.read_file(source_path)
    # img.show()
    # print("size", img.size)
    #
    # extention = "dcm"
    # is2D = False
    # view = "saggital"
    # #source_path = "C:\\Users\\NM\\PycharmProjects\\dicom_AD_test_data\\dicom_AD_test_data\\ct_AD_ID_001_1.2.410.200055.998.998.1707237463.28700.1526446844.649.dcm"
    # source_path = "C:\\Users\\NM\\PycharmProjects\\dicom_AD_test_data\\AD_sub1"
    # idio = ImageDataIO(extention, is2D, view)
    # img = idio.read_file(source_path)#img.show()
    # for ind in range(len(img)):
    #
    #     img[ind].show()
    # print("size", img[0].size)

    # source_path = "C:\\Users\\NM\\PycharmProjects\\MDBManager\\test\\anonymous_test_data\\1\\cw384302_1lr.nii"
    # extention = "nii"
    # is2D = False
    # view = "coronal"
    # idio = ImageDataIO(extention, is2D, view)
    # img = idio.read_file(source_path)  # img.show()
    # img[0].show()
    # print("size", img[0].size)

    view = "axial"
    data_dir_child_path = [
        '/media/miruware/b7386651-c7ef-4c44-a755-46ddd1e945ac/sylee/ModelComparison/dataset/raw_PET_dcm_sameas180903/gr1',
        '/media/miruware/b7386651-c7ef-4c44-a755-46ddd1e945ac/sylee/ModelComparison/dataset/raw_PET_dcm_sameas180903/gr2',
        '/media/miruware/b7386651-c7ef-4c44-a755-46ddd1e945ac/sylee/ModelComparison/dataset/raw_PET_dcm_sameas180903/gr3']
    for ind, class_dir in enumerate(data_dir_child_path):  # gr1, gr2, gr3
        idio = ImageDataIO("dcm", dataDim="2D", instanceIsOneFile=False, modeling="3D", view=view)
        datas, filenames = idio.read_files_from_dir(class_dir)  # (N, C, H, W)
        # data : list of dcm file name, filename : folder(pid) list

        print("data shape", datas.shape)
        # 8-bit img
        # datas = idio.convertUint8(datas)
        # datas = idio._resizing_channel(datas, resize_size=None, channel_size=1)
        # datas = idio._cropping(datas, (100, 100, 300, 300))

        # for ind2, fname in tqdm.tqdm(enumerate(filenames)):
        #     save_path = os.path.join("/media/miruware/b7386651-c7ef-4c44-a755-46ddd1e945ac/sylee/ModelComparison/Histogram/raw_img_histogram", "gr" + str(ind + 1), str(fname) + ".png")
        #     idio.histogram_img(datas[ind2], save_path)

