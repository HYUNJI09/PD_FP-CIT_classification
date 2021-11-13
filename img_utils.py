import numpy as np
import math
import matplotlib.pyplot as plt

from custom_exception import CustomException
from ImageDataIO import ImageDataIO
from Interpolation3D import ImageInterpolator
# thresholding with cutting ratio between max / min values of each of matrix
# input : matrix to mask, standard matrix, cutting ratio
# output : masked matrix
def thresholding_matrix(mat_to_mask, c_ratio, std_mat=None):
    """
    :param mat_to_mask: this param can get both 2D image (H, W, C) or 3D image (D, H, W, C)
    :param c_ratio: float32
    :param std_mat: this param can get both 2D image (H, W, C) or 3D image (D, H, W, C)
    :return: masked matrix
    """
    # 2D
    #(H, W), (H, W, 1)
    #(H, W, 3)

    # 3D
    # (D, H, W), (D, H, W, 1)
    # (D, H, W, 3)
    if c_ratio is None:
        return mat_to_mask

    if std_mat is None:
        std_mat = mat_to_mask

    mat_to_mask = np.array(mat_to_mask)

    if len(mat_to_mask.shape) < 3 or len(std_mat.shape)< 3 or len(mat_to_mask.shape) > 4 or len(std_mat.shape) > 4:
        CustomException("[!] This is not allowed")

    std_mat = np.array(std_mat)
    # when both mat_to_mask and std_mat are 2D or 3D
    if len(mat_to_mask.shape) == len(std_mat.shape):
        threshold = std_mat.min() + (std_mat.max() - std_mat.min()) * c_ratio  # (D, H, W)
        bool_mask = std_mat > threshold  # region to show
        int_mask = bool_mask.astype(np.uint8)

        masked = np.ma.masked_where(int_mask == 0, int_mask)

        return mat_to_mask * masked
    # when mat_to_mask is 3D and std_mat are 2D
    elif len(mat_to_mask.shape) > len(std_mat.shape):

        threshold = std_mat.min() + (std_mat.max() - std_mat.min()) * c_ratio  # (D, H, W)
        bool_mask = std_mat > threshold  # region to show
        int_mask = bool_mask.astype(np.uint8)

        masked = np.ma.masked_where(int_mask==0, int_mask)

        color_masked = np.array([masked, masked, masked])  # (3, D, H, W)
        color_masked = np.transpose(color_masked, axes=(1, 2, 3, 0))

        return mat_to_mask * color_masked
    # when mat_to_mask is 2D and std_mat are 3D
    elif len(mat_to_mask).shape < len(std_mat.shape):
        CustomException("[!] This is not allowed")

def get_nii_numpy_from_dir(nii_dir_path, target_size=None):
    idio = ImageDataIO(extention="nii", dataDim="3D", instanceIsOneFile=True, modeling=None, view="axial")
    imgs, filenames = idio.read_files_from_dir(nii_dir_path)
    imgs = idio.convert_PIL_to_numpy(imgs)
    print("debug", imgs.shape)
    img_shape = imgs[0].shape
    if target_size is None:
        target_size = img_shape


    imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=target_size)
    resized_imgs = np.array([imgip.interpolateImage(img) for img in imgs])
    return resized_imgs, filenames

def level_set(img, cutoff=100, d_ratio = 0.1):
    img = np.array(img)
    b_top = img.max()
    bottom = img.min()
    _range = b_top - bottom
    d_height = _range*d_ratio

    _level = b_top
    while _level > bottom:
        coords = np.where(img>_level)
        num_pixels = len(coords[0])
        if num_pixels > cutoff:
            return _level/b_top, num_pixels
        else:
            _level -= d_height

    return None

def draw_img_on_grid_v2(imgs_list, save_path = None, is2D=False, _input_img_alpha=None, _overlap_img_alpha=None,_title=None,
                     _input_img_cmap=None, _overlap_img_cmap=None, std_mat=None, c_ratio_list=None, level_set_cutoff=None, level_set_d_ratio=0.01):
    """
    :param imgs_list: this variable's shape is [img1(N, H, W, C), img2(N, H, W, C)]
    :param save_path:
    :param is2D:
    :param _input_img_alpha:
    :param _overlap_img_alpha:
    :param _title:
    :param _input_img_cmap:
    :param _overlap_img_cmap:
    :return:
    """
    if is2D:
        plt.imshow(imgs_list)
        plt.show()
    else:
        #tmp_img = np.array(imgs_list[0])
        img_shape = imgs_list[0].shape
        num_imgs = img_shape[0]
        grid_height = img_shape[1]
        grid_width = img_shape[2]
        # print("debug ", len(imgs_list[0]))
        # print("debug ", len(imgs_list[1]))
        board_size = math.ceil(math.sqrt(num_imgs))
        r = c = board_size

        # input: imgs_list, board_size
        # return : grid_board_list
        grid_board_list = [] # (N, r*grid_height, c*grid_width)
        for ind, imgs in enumerate(imgs_list): # imgs' shape,  (N, H, W, C)
            grid_board = np.zeros((r*grid_height, c*grid_width))

            img_ind = 0
            for r_ind in range(r):
                for c_ind in range(c):
                    grid_board[r_ind*grid_height:r_ind*grid_height+grid_height, c_ind*grid_width:c_ind*grid_width+grid_width] = imgs[img_ind]
                    img_ind+=1
                    if img_ind >= num_imgs:
                        break
                if img_ind >= num_imgs:
                    break


            grid_board_list.append(grid_board)

        if _title is not None:
            fig = plt.figure()
            fig.suptitle(_title)
        for overlap_ind in range(len(grid_board_list)):
            tmp_img = grid_board_list[overlap_ind]
            #print("imgs_list[overlap_ind][img_ind]", np.array(imgs_list[overlap_ind][img_ind]).shape) # (32, 32, 1)

            if c_ratio_list is not None:
                tmp_img = thresholding_matrix(mat_to_mask=tmp_img, std_mat=std_mat,
                                                           c_ratio=c_ratio_list[overlap_ind])
            #print("debug, tmp_img,  in draw_img_on_grid_v2", tmp_img.shape) # (776, 632)

            if overlap_ind == 0:
                # tmp_img_min = tmp_img.min()
                # tmp_img_range = tmp_img.max() - tmp_img.min()
                # tmp_img = (tmp_img - tmp_img_min) / tmp_img_range
                # tmp_img[0, 0] = 0
                # tmp_img[0, 1] = 255

                plt.imshow(tmp_img , cmap=_input_img_cmap, alpha=_input_img_alpha)
            elif overlap_ind != 0:
                if level_set_cutoff is not None:
                    _threshold, num_pixels = level_set(tmp_img, cutoff=level_set_cutoff, d_ratio=level_set_d_ratio)
                    print("_threshold, num_pixels", _threshold, num_pixels)
                    tmp_img = thresholding_matrix(mat_to_mask=tmp_img, std_mat=std_mat,
                                                  c_ratio=_threshold)
                plt.imshow(tmp_img, cmap=_overlap_img_cmap, alpha=_overlap_img_alpha)


        if save_path is None:
            plt.show()
        else:
            plt.axis('off')
            #plt.savefig(save_path, bbox_inches= 'tight', pad_inches = 0)
            plt.savefig(save_path)
    plt.close("all")

    return

def draw_img_on_grid(imgs_list, save_path = None, is2D=False, _input_img_alpha=None, _overlap_img_alpha=None,_title=None,
                     _input_img_cmap=None, _overlap_img_cmap=None):
    """

    :param imgs_list: this variable's shape is [img1(N, H, W, C), img2(N, H, W, C)]
    :param save_path:
    :param is2D:
    :param _input_img_alpha:
    :param _overlap_img_alpha:
    :param _title:
    :param _input_img_cmap:
    :param _overlap_img_cmap:
    :return:
    """
    if is2D:
        plt.imshow(imgs_list)
        plt.show()
    else:
        num_imgs = len(imgs_list[0])
        # print("debug ", len(imgs_list[0]))
        # print("debug ", len(imgs_list[1]))
        board_size = math.ceil(math.sqrt(num_imgs))
        r = c = board_size
        if _title is not None:
            fig = plt.figure()
            fig.suptitle(_title)
        img_ind = 0
        for r_ind in range(r):
            for c_ind in range(c):
                c_ind_ = c_ind + 1
                plt.subplot(r, c, r_ind * c + c_ind_)
                #print("debug img_ind", img_ind)
                for overlap_ind in range(len(imgs_list)):
                    #print("imgs_list[overlap_ind][img_ind]", np.array(imgs_list[overlap_ind][img_ind]).shape) # (32, 32, 1)
                    if overlap_ind == 0:
                        plt.imshow(imgs_list[overlap_ind][img_ind], cmap=_input_img_cmap, alpha=_input_img_alpha)
                    elif overlap_ind != 0:
                        plt.imshow(imgs_list[overlap_ind][img_ind], cmap=_overlap_img_cmap, alpha=_overlap_img_alpha)

                img_ind = img_ind + 1
                if img_ind >= num_imgs:
                    if save_path is None:
                        plt.show()
                    else:
                        plt.savefig(save_path)
    plt.close("all")

    return