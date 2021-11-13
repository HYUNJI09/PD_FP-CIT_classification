import numpy as np
from scipy.interpolate import RegularGridInterpolator

# customized module
from custom_exception import *
from ImageDataIO import ImageDataIO

class ImageInterpolator():
    """
    This class only work for 3 kind of modes
    "G2D" - one channel, 2D image (H, W) or (H, W, 1)?
    "C2D" - 3 color channel, 2D image (H, W, 3)
    "G3D" - one channel, 3D image (D, H, W) or (D, H, W, 1)?
    """
    def __init__(self, is2D, num_channel, target_size):
        self._is2D = is2D
        self._num_channel = num_channel
        self._target_size = target_size
        if self._is2D and len(self._target_size) <=2 :
            raise CustomException("target_size variable must be larger then 2 when input data is on 2-Dimension")
        if self._is2D and self._num_channel == 1:
            self._mode = "G2D"
        elif self._is2D and self._num_channel == 3:
            self._mode = "C2D"
        elif not self._is2D and self._num_channel ==1:
            self._mode = "G3D"
        else :
            raise CustomException("not supported mode")

    def _make_linspace(self, input_data):
        input_data = np.array(input_data)
        self._img_size = input_data.shape # # tuple of length for each axis

        if self._mode == "G2D":
            self._h_linspace = np.linspace(0, self._img_size[0]-1, self._img_size[0])
            self._w_linspace = np.linspace(0, self._img_size[1]-1, self._img_size[1])
            # self._meshgrid = np.meshgrid(self._h_linspace, self._w_linspace, indexing='ij', sparse=True)

        elif self._mode == "C2D":
            self._h_linspace = np.linspace(0, self._img_size[0] - 1, self._img_size[0])
            self._w_linspace = np.linspace(0, self._img_size[1] - 1, self._img_size[1])
            self._c_linspace = np.linspace(0, self._img_size[2] - 1, self._img_size[2])
            # self._meshgrid = np.meshgrid(self._h_linspace, self._w_linspace, self._c_linspace, indexing='ij', sparse=True)
        elif self._mode == "G3D":
            self._d_linspace = np.linspace(0, self._img_size[0] - 1, self._img_size[0])
            self._h_linspace = np.linspace(0, self._img_size[1] - 1, self._img_size[1])
            self._w_linspace = np.linspace(0, self._img_size[2] - 1, self._img_size[2])
            # self._meshgrid = np.meshgrid(self._d_linspace, self._h_linspace, self._w_linspace, indexing='ij',
            #                              sparse=True)

    def _get_interpolator(self, input_image):
        if self._mode == "G2D":
            h = self._h_linspace
            w = self._w_linspace
            self._interpolator = RegularGridInterpolator((h, w), input_image)
        elif self._mode == "C2D":
            h = self._h_linspace
            w = self._w_linspace
            c = self._c_linspace
            self._interpolator = RegularGridInterpolator((h, w, c), input_image)
        elif self._mode == "G3D":
            d = self._d_linspace
            h = self._h_linspace
            w = self._w_linspace
            self._interpolator = RegularGridInterpolator((d, h, w), input_image)

    def interpolateImage(self, input_image):
        self._make_linspace(input_image)
        self._get_interpolator(input_image)

        if self._mode == "G2D":
            new_h_linspace = np.linspace(0, self._img_size[0]-1, self._target_size[0])
            new_w_linspace = np.linspace(0, self._img_size[1]-1, self._target_size[1])

            new_h = np.stack([new_h_linspace]*self._target_size[1], axis=1)
            new_w = np.stack([new_w_linspace] * self._target_size[0], axis=0)
            #print("debug", new_h_linspace.shape)
            # new_h, new_w = np.meshgrid(new_h_linspace, new_w_linspace)
            # print("debug", new_h.shape)
            # #target_image_grid = np.array(list(zip(new_h.flatten(), new_w.flatten()))).reshape([self._target_size[0], self._target_size[1], 2])
            new_h = new_h[:,:,None]
            new_w = new_w[:, :, None]
            # print("debug", new_h.shape, new_w.shape)
            target_image_grid = np.concatenate([new_h, new_w], axis=-1)
            #print("debug", target_image_grid.shape)
        elif self._mode == "C2D":
            new_h_linspace = np.linspace(0, self._img_size[0] - 1, self._target_size[0])
            new_w_linspace = np.linspace(0, self._img_size[1] - 1, self._target_size[1])
            new_c_linspace = self._c_linspace

            new_h = np.stack([new_h_linspace] * self._target_size[1], axis=1)
            new_h = np.stack([new_h] * self._target_size[2], axis=2)

            new_w = np.stack([new_w_linspace] * self._target_size[0], axis=0)
            new_w = np.stack([new_w] * self._target_size[2], axis=2)

            new_c = np.stack([new_c_linspace] * self._target_size[0], axis=0)
            new_c = np.stack([new_c] * self._target_size[1], axis=1)

            new_h = new_h[:, :, :, None]
            new_w = new_w[:, :, :, None]
            new_c = new_c[:, :, :, None]
            # print("debug", new_h.shape, new_w.shape)
            target_image_grid = np.concatenate([new_h, new_w, new_c], axis=-1)
            #print("debug", target_image_grid.shape)
        elif self._mode == "G3D":
            new_d_linspace = np.linspace(0, self._img_size[0] - 1, self._target_size[0])
            new_h_linspace = np.linspace(0, self._img_size[1] - 1, self._target_size[1])
            new_w_linspace = np.linspace(0, self._img_size[2] - 1, self._target_size[2])

            new_d = np.stack([new_d_linspace] * self._target_size[1], axis=1)
            new_d = np.stack([new_d] * self._target_size[2], axis=2)

            new_h = np.stack([new_h_linspace] * self._target_size[0], axis=0)
            new_h = np.stack([new_h] * self._target_size[2], axis=2)

            new_w = np.stack([new_w_linspace] * self._target_size[0], axis=0)
            new_w = np.stack([new_w] * self._target_size[1], axis=1)

            new_d = new_d[:, :, :, None]
            new_h = new_h[:, :, :, None]
            new_w = new_w[:, :, :, None]
            # print("debug", new_h.shape, new_w.shape)
            target_image_grid = np.concatenate([new_d, new_h, new_w], axis=-1)
            #print("debug", target_image_grid.shape)

        return self._interpolator(target_image_grid)

if __name__ == "__main__":
    #target_dir = "C:\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\FBB_8\\nii(processed)\\pure group test\\nc_ad\\0"
    target_dir =  "C:\\Users\\NM\\PycharmProjects\\NeurologyClinicalStudy\\dataset\\PET\\dataset\\bapl2\\gr2\\2"
    idio = ImageDataIO(extention="nii", is2D=False, view="axial")
    img, _ = idio.read_files_from_dir(target_dir)  # img.show()
    print(img)
    img = idio.convert_PIL_to_numpy(img)
    #print("img shape", img.shape)
    print(img[0])
    idio.show_one_img(img[0])
    imgip = ImageInterpolator(is2D=False, num_channel=1, target_size=(64, 64, 64))
    resized_img = imgip.interpolateImage(img[0])
    idio.show_one_img(resized_img)

# def f(x, y, z):
#     return 2*x**3 + 3*y**2 - z
# 
# x = np.linspace(1, 4, 11) # 3
# y = np.linspace(4, 7, 22) # 3
# z = np.linspace(7, 9, 33) # 2
# print(x)
# print(y)
# print(z)
# 
# print("x len", len(x))
# print("y len", len(y))
# print("z len", len(z))
# 
# data = f(*np.meshgrid(x, y, z, indexing='ij', sparse=True))
# print("data", data) # data[i,j,k] = f(x[i], y[j], z[k])
# #print(np.meshgrid(x, y, z, indexing='ij', sparse=True))
#print(np.linspace(0, 3, 10))
# print(np.meshgrid([0,1],[0,1,2]))