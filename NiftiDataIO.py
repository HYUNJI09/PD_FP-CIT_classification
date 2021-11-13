import os
import sys
import math
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from nibabel import load as nib_load
from custom_exception import *


class NiftiDataIO():
    # arrange dimension for each type of view (axial / saggital / coronal)
    def __init__(self, dataDimInOneFile, instanceIsOneFile=True, view=None):
        """

        :param dataDim: data dim to treat, '2D' of image or '3D' of image, in a file;
        :param instanceIsOneFile: how is an instance stored for one instance; True'one_file' / 'multiple_files'
        :param modeling: how to handle an instance, e.g 2D or 3D
        :param view:
        """

        self._dataDimInOneFile = dataDimInOneFile
        self._instanceIsOneFile = instanceIsOneFile
        self._view = view
        return

    # handling one file
    def read_file(self, source_path):
       return self._read_nifti_img(source_path)

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
        return file_list, filename_list # list of PIL Image obj

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
        nib_img = nib_load(source_path)
        nib_img = np.array(nib_img.get_data())
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

            nii_imgs = nii_imgs.astype(np.uint8)
            nii_imgs = np.transpose(nii_imgs, [2, 0, 1])
            return [Image.fromarray(nii_img) for nii_img in nii_imgs]
        elif self._dataDimInOneFile == "3D" and self._instanceIsOneFile:
            if self._view == "axial" or self._view == "transaxial":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 1))  # (95, 79, 68)
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
                nib_img = self._change_image_intensity_range(nib_img)
                nib_img = nib_img.astype(np.uint8)
                nib_img = np.transpose(nib_img, [2, 0, 1])
                # print("3D shape", nib_img.shape)
                return [Image.fromarray(img) for img in nib_img]

            else:
                raise CustomException(
                    "the state that _instanceIsOneFile is False is not defined yet in Nifti type")

        # when "4D" data, (79, 95, 68, 27)
        elif self._dataDimInOneFile == "4D" and self._instanceIsOneFile:
            if self._view == "axial" or self._view == "transaxial":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 1))  # (95, 79, 68, 27)
            elif self._view == "saggital":
                nib_img = np.rot90(nib_img, k=1, axes=(0, 2))  # (68, 95, 79, 27)
            elif self._view == "coronal":
                nib_img = np.rot90(nib_img, k=1, axes=(1, 2))  # (79, 68, 95, 27)
                nib_img = np.rot90(nib_img, k=3, axes=(0, 1))  # (68, 79, 68, 27)
            nib_img = self._change_image_intensity_range(nib_img)
            nib_img = nib_img.astype(np.uint8)


            # to iter image on frame first, (T, D, H, W)
            nib_img = np.transpose(nib_img, [3, 2, 0, 1]) # (27 68, 95, 79)

            _pil_obj_list = []
            # for time-frames
            for _t_frame in nib_img:
                _depth_pil_obj_list = []
                # for depth
                for _img in _t_frame:
                    _depth_pil_obj_list.append(Image.fromarray(_img))
                _pil_obj_list.append(_depth_pil_obj_list)

            # print("3D shape", nib_img.shape)
            return _pil_obj_list
        else:
            raise CustomException(
                "the state that _instanceIsOneFile is False is not defined yet in Nifti type")

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

    def _resizing_channel(self, img_obj, resize_size, channel_size=None):
        """
        :param img_obj: img_obj have to be a list of images, which is "2D", "3D", or "4D"
        :param resize_size:
        :param channel_size:
        :return:
        """

        if sys.getsizeof(img_obj) == 0:
            print("size of img_data object is 0")
            return None
        # if isinstance(img_obj, list):
        #     img_shape = img_obj[0].size
        # else :
        #     img_shape = img_obj.size

        # resize
        if resize_size is not None:
            if self._dataDimInOneFile == "3D": # (N, D, H, W, C)
                sub_list = []
                for sub in img_obj :
                    imgs = [img.resize(resize_size) for img in sub]
                    sub_list.append(imgs)
                img_obj = sub_list

            elif self._dataDimInOneFile == "2D": # (N, H, W, C)
                #img_data = np.array([cv2.resize(np.array(x), resize_size) for x in img_data]).astype(np.float32)
                if isinstance(img_obj, list):
                    img_obj = [img.resize(resize_size) for img in img_obj]
                else :
                    img_obj = img_obj.resize(resize_size)
            elif self._dataDimInOneFile == "4D": # (N, T, D, H, W, C)
                obj_list = []
                for _multi_vols in img_obj:
                    multi_vol_list = []
                    for _t_frame in _multi_vols:
                        _img_list = []
                        for _img in _t_frame:
                            _img_list.append(_img.resize(resize_size))
                        multi_vol_list.append(_img_list)
                    obj_list.append(multi_vol_list)
                    img_obj = obj_list


        # check channel
        if self._dataDimInOneFile == "3D" :
            sub_list = []
            for sub in img_obj:
                if channel_size is not None and channel_size == 1:
                    imgs = [img.convert("L") for img in sub]
                elif channel_size is not None and channel_size == 3:
                    imgs = [img.convert("RGB") for img in sub]
                elif channel_size is not None and channel_size == 4:
                    imgs = [img.convert("RGBA") for img in sub]
                sub_list.append(imgs)
            img_obj = sub_list
        elif self._dataDimInOneFile == "2D" : #isinstance(img_obj, list):
            if channel_size is not None and channel_size == 1:
                img_obj = [img.convert("L") for img in img_obj]
            elif channel_size is not None and channel_size == 3:
                img_obj = [img.convert("RGB") for img in img_obj]
            elif channel_size is not None and channel_size == 4:
                img_obj = [img.convert("RGBA") for img in img_obj]
        elif self._dataDimInOneFile == "4D" :
            obj_list = []
            for sub in img_obj:
                _3d_img_list = []
                for _3d_img in sub:
                    if channel_size is not None and channel_size == 1:
                        imgs = [img.convert("L") for img in _3d_img]
                    elif channel_size is not None and channel_size == 3:
                        imgs = [img.convert("RGB") for img in _3d_img]
                    elif channel_size is not None and channel_size == 4:
                        imgs = [img.convert("RGBA") for img in _3d_img]
                    _3d_img_list.append(imgs)
                obj_list.append(_3d_img_list)
            img_obj = obj_list

        else: # img_obj is just one file
            if channel_size is not None and channel_size == 1:
                img_obj = img_obj.convert("L")
            elif channel_size is not None and channel_size == 3:
                img_obj = img_obj.convert("RGB")
            elif channel_size is not None and channel_size == 4:
                img_obj = img_obj.convert("RGBA")

        return img_obj

    def convert_PIL_to_numpy(self, img_obj):

        if isinstance(img_obj, list):
            if self._dataDimInOneFile == "3D":
                sub_list = []
                for sub in img_obj:
                    img_list = []
                    for img in sub:
                        #print(np.asarray(img))
                        img_list.append(np.asarray(img))
                    img_list = np.array(img_list)
                    sub_list.append(img_list)
                return np.array(sub_list)

            elif self._dataDimInOneFile == "2D":
                return np.array([np.array(img) for img in img_obj])

            elif self._dataDimInOneFile == "4D":
                obj_list = [] # 5D
                for sub in img_obj: # 5D, (N, T, D, H, W, 1)
                    _3d_list = [] # 4D
                    for _3d_img in sub:
                        _2d_img_list = [] # 4D
                        for _img in _3d_img:
                            _2d_img_list.append(np.asarray(_img)) # 2D

                        _3d_list.append(np.array(_2d_img_list))
                    obj_list.append(np.array(_3d_list))


                return np.array(obj_list)

        else: # img_obj is just one instance not list
            return np.asarray(img_obj)

if __name__ == "__main__":
    print("hello")
    source_dir = r"C:\Users\hkang\MatlabProjects\SPM\spm12\spm12\custom_modules\Dy_PET_SUV\total_dy_set\4_count_match_center_scale_PET_v4"
    source_filepath = r"count_match_2382_1.3.12.2.1107.5.1.4.11002.30000018091402105889900020259.nii"
    target_filepath = os.path.join(source_dir, source_filepath)

    # nib_img = nib_load(target_filepath)
    # print(nib_img)
    # nib_img = np.array(nib_img.get_data())
    # print(nib_img.shape)
    ndio = NiftiDataIO("4D", view="axial")
    nib_img = ndio.read_file(target_filepath)
    print(nib_img)

    nib_img = ndio._resizing_channel([nib_img], (128, 128), channel_size=1)
    print(len(nib_img))


    nib_img = ndio.convert_PIL_to_numpy(nib_img)
    print(nib_img.shape)

    # nib_imgs = ndio.read_files_from_dir(source_dir)
    #     # print(nib_imgs)