import os
import gc

import numpy as np
import pandas as pd

from ImageDataIO import *
from Interpolation3D import ImageInterpolator
from preprocess import apply_min_max_normalization
class NIIDataIO():
    def __init__(self):
        return

    # def load(self, data_dir, extension, dataDim, instanceIsOneFile, data_dir_child, channel_size=None, view=None):
    #     """
    #             :param data_dir:
    #             :param extension:
    #             :param is2D:
    #             :param view:
    #             :param data_dir_child: "labeled" or "sample", argument "labeled" means data_dir is the parent directory including labeled directory of sample files
    #                                    argument "sample" means data_dir consists of a set of sample datas(files) directly
    #             :return:
    #             """
    #     # ????
    #     if data_dir_child == "labeled":
    #         child_dirs = os.listdir(data_dir)
    #         self._num_classes = len(child_dirs)
    #         child_dirs.sort()
    #         data_dir_child_path = [os.path.join(data_dir, child) for child in child_dirs]
    #         print("data_dir_child_path", data_dir_child_path)
    #     elif data_dir_child == "sample":
    #         data_dir_child_path = [data_dir]
    #         print("data_dir_child_path", data_dir_child_path)
    #     elif data_dir_child == None:
    #         data_dir_child_path = [data_dir]
    #         print("load3DImage -> data_dir_child is None")
    #         print("data_dir_child_path", data_dir_child_path)
    #         # CustomException("Argument data_dir_child need to be defined!")
    #
    #     self._3D_data = []
    #     self._3D_data_filename = []
    #     self._label = []
    #     self._label_name = []
    #     self._class_name = []
    #     for ind, class_dir in enumerate(data_dir_child_path):
    #         print("debug load3DImage, read class_dir data", ind, class_dir)
    #         # img_data_path_list = [os.path.join(class_dir, img_data) for img_data in os.listdir(class_dir)]
    #         # print("debug", img_data_path_list)
    #         self._class_name.append(os.path.basename(class_dir))
    #         idio = ImageDataIO(extension, dataDim=dataDim, instanceIsOneFile=instanceIsOneFile, modeling="3D",
    #                            view=view)
    #         _data, _filename = idio.read_files_from_dir(class_dir)
    #         #print("[debug] ##############", len(np.unique(_data))) # [debug] ############## 6
    #
    #         try:
    #             _scaled_data = apply_min_max_normalization(_data)
    #         except IndexError as ie:
    #             shape_list = []
    #             for data_elem in _data:
    #                 shape_ = np.array(data_elem).shape
    #                 shape_list.append(str(shape_))
    #
    #             result_df = pd.DataFrame({'img_shape': shape_list, 'filename': _filename})
    #             result_df.to_excel(r'/root/PycharmProjects/FPCIT_PD_2021/img_shape.xlsx')
    #             return
    #         #print("[debug] ##############", len(np.unique(_scaled_data))) # 11
    #
    #         _scaled_data = _scaled_data*255
    #
    #         _data = idio.convert_numpy_to_PIL(_scaled_data)
    #         _data = idio._resizing_channel(_data, resize_size=None, channel_size=channel_size)
    #
    #         _data_len = len(_data)
    #
    #         # train_bapl1 = np.array([np.array(train_bapl1_) for train_bapl1_ in train_bapl1])
    #         _data = idio.convert_PIL_to_numpy(_data)
    #         print(os.path.basename(class_dir), "class", _data.shape)
    #         self._3D_data.append(_data)
    #         self._3D_data_filename.append(_filename)
    #         self._label.append([ind] * _data_len)
    #         self._label_name.append([os.path.basename(class_dir)] * _data_len)
    #     self._3D_data = np.concatenate(self._3D_data)
    #     self._3D_data_filename = np.concatenate(self._3D_data_filename)
    #     self._label = np.concatenate(self._label)
    #     self._label_name = np.concatenate(self._label_name)
    #
    #     return

    def load_v2(self, data_dir, extension, dataDim, instanceIsOneFile, data_dir_child, channel_size = None, view = None):
        """
        especially this code include min-max normalization and reduce input image from (400, 400) to (200, 200)
        :param data_dir:
        :param extension:
        :param is2D:
        :param view:
        :param data_dir_child: "labeled" or "sample", argument "labeled" means data_dir is the parent directory including labeled directory of sample files
                               argument "sample" means data_dir consists of a set of sample datas(files) directly
        :return:
        """
        # ????
        if data_dir_child == "labeled":
            child_dirs = os.listdir(data_dir)

            self._num_classes = len(child_dirs)
            child_dirs.sort()
            data_dir_child_path = [os.path.join(data_dir, child) for child in child_dirs]
            print("data_dir_child_path", data_dir_child_path)
        elif data_dir_child == "sample":
            data_dir_child_path = [data_dir]
            print("data_dir_child_path", data_dir_child_path)
        elif data_dir_child == None:
            data_dir_child_path = [data_dir]
            print("load3DImage -> data_dir_child is None")
            print("data_dir_child_path", data_dir_child_path)
            # CustomException("Argument data_dir_child need to be defined!")

        self._3D_data = []
        self._3D_data_filename = []
        self._label = []
        self._label_name = []
        self._class_name = []
        for ind, class_dir in enumerate(data_dir_child_path):
            print("debug load3DImage, read class_dir data", ind, class_dir)
            # img_data_path_list = [os.path.join(class_dir, img_data) for img_data in os.listdir(class_dir)]
            # print("debug", img_data_path_list)
            self._class_name.append(os.path.basename(class_dir))
            idio = ImageDataIO(extension, dataDim=dataDim, instanceIsOneFile=instanceIsOneFile, modeling="3D",
                               view=view)

            shape_list = []

            #_data, _filename = idio.read_files_from_dir(class_dir)\
            dir_data_list = []
            filename_list = []
            for filename in os.listdir(class_dir):
                _data = idio.read_file(os.path.join(class_dir, filename))
                #_data = _data[:,99:299, 99:299] # (110, 400, 400) -> (110, 200, 200) # slicing
                _min = _data.min()
                _max = _data.max()
                _norm_data = (_data-_min) / (_max-_min)

                # idio.show_one_img_v3(_norm_data, is2D=False, cmap=plt.get_cmap('gray'))
                # print("max", _norm_data.max(), "min", _norm_data.min())
                # print(_norm_data.shape)
                dir_data_list.append(_norm_data)
                filename_list.append(filename)

                # shape_ = np.array(_norm_data).shape
                # shape_list.append(str(shape_))
            # result_df = pd.DataFrame({'img_shape': shape_list, 'filename': filename_list})
            # result_df.to_excel(r'/home/ubuntu/hjshin/2021_FPCIT_PET_Early_Static/img_shape_'+str(ind)+'.xlsx')


            dir_data_list = np.array(dir_data_list)
            _filename = np.array(filename_list)

            # try:
            #     _scaled_data = apply_min_max_normalization(_data)
            #     _scaled_data = _scaled_data * 255
            #
            #
            # except IndexError as ie:
            #     shape_list = []
            #     for data_elem in _data:
            #         shape_ = np.array(data_elem).shape
            #         shape_list.append(str(shape_))
            #
            #     result_df = pd.DataFrame({'img_shape': shape_list, 'filename': _filename})
            #     result_df.to_excel(r'/media/ubuntu/40d020bc-904f-49b4-ac95-2cfbaec96c2a/img_shape.xlsx')
            #     return
            #print("[debug] ##############", len(np.unique(_scaled_data))) # 11

            dir_data_list = np.array(dir_data_list)*255

            _data = idio.convert_numpy_to_PIL(dir_data_list)
            _data = idio._resizing_channel(_data, resize_size=None, channel_size=channel_size)

            _data_len = len(_data)

            # train_bapl1 = np.array([np.array(train_bapl1_) for train_bapl1_ in train_bapl1])
            _data = idio.convert_PIL_to_numpy(_data)
            print(os.path.basename(class_dir), "class", _data.shape)
            self._3D_data.append(_data)
            self._3D_data_filename.append(_filename)
            self._label.append([ind] * _data_len)
            self._label_name.append([os.path.basename(class_dir)] * _data_len)
        self._3D_data = np.concatenate(self._3D_data, axis=0)
        self._3D_data_filename = np.concatenate(self._3D_data_filename)
        self._label = np.concatenate(self._label)
        self._label_name = np.concatenate(self._label_name)

        return


    def resize(self, input_shape):
        imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=input_shape)
        self._3D_data = [imgip.interpolateImage(img) for img in self._3D_data]
        if self._3D_data[0].shape[-1] != 1 or self._3D_data[0].shape[-1] != 3:
            self._3D_data = np.array(self._3D_data)[:, :, :, :, None]
        return

    def show_one_img_v3(self, img, is2D=True, cmap=None, save_path = None):
        """

        :param imgs:
        :param is2D: (X, Y) or (D, X, Y) ; not tested when img have C yet
        :param cmap:
        :return:
        """
        if is2D : # one 2D image
            plt.imshow(img, cmap)
            if save_path is not None:
                plt.savefig(save_path)
            else:
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
                        if save_path is not None:
                            plt.savefig(save_path)
                        else:
                            plt.show()
                        plt.clf()
                        return

        return

if __name__ == "__main__":
    print("test")
    data_dir = r"/media/ubuntu/40d020bc-904f-49b4-ac95-2cfbaec96c2a/210522_NC_PD_nPD"
    extension = "nii"
    dataDim = "3D"  # 3D
    view = "axial"

    input_shape = (64, 64, 64)
    channel_size = 1

    nii_io = NIIDataIO()
    nii_io.load_v2(data_dir, extension=extension, data_dir_child="labeled",
                            dataDim=dataDim, instanceIsOneFile=True, channel_size=channel_size, view=view)

    print(np.array(nii_io._3D_data).shape)  # (213, 68, 95, 79, 1)
    print(np.array(nii_io._3D_data_filename).shape)  # (256,)
    print(np.array(nii_io._label).shape)  # (256,)
    print(np.array(nii_io._label_name).shape)  # (256,)
    print(np.array(nii_io._class_name))  # ['0_bapl1' '1_bapl23']

    #nii_io.resize((64, 64, 64))
    #print(np.array(nii_io._3D_data).shape)  # (256, 64, 64, 64, 1)





