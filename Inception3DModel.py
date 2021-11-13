from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense, Concatenate, Lambda
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model
from keras.activations import softmax
from keras_applications import get_submodules_from_kwargs
from keras_applications.imagenet_utils import _obtain_input_shape


class Inception3D():
    def __init__(self, input_size):
        self._input_size = input_size
        self._bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

    def conv3d_bn(self, x, filters, filter_size, strides=(1, 1, 1), padding = 'same', name=None):
        if name is not None:
            bn_name = "conv3d/bn"
            activate_name = "conv3d/relu"
        else :
            bn_name = None
            activate_name = None

        if K.image_data_format() == 'channels_first':
            self._bn_axis = 1
        else:
            self._bn_axis = 4
        x = Conv3D(filters, filter_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization(axis=self._bn_axis, scale=False, name=bn_name)(x)
        # x = BatchNormalization(
        #     axis=self._bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = Activation('relu', name=activate_name)(x)
        return x

    def inception3d(self, num_classes = 2, pooling = 'avg'):

        input = Input(shape=self._input_size)

        if K.image_data_format() == 'channels_first': # (B, C, H, W, D)
            channel_axis = 1
        else:
            channel_axis = 4

        x = self.conv3d_bn(input, 32, 3, strides=(2, 2, 2), padding='valid')
        x = self.conv3d_bn(x, 32, 3, padding='valid')
        x = self.conv3d_bn(x, 64, 3, padding='valid')
        x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

        x = self.conv3d_bn(x, 80, 1, padding='valid')
        x = self.conv3d_bn(x, 192, 3, padding='valid')
        # x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)


        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self.conv3d_bn(x, 64, 1)

        branch5x5 = self.conv3d_bn(x, 48, 1)
        branch5x5 = self.conv3d_bn(branch5x5, 64, 5)

        branch3x3dbl = self.conv3d_bn(x, 64, 1)
        branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, 3)
        branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, 3)

        branch_pool = AveragePooling3D((3, 3, 3),
                                              strides=(1, 1, 1),
                                              padding='same')(x)
        branch_pool = self.conv3d_bn(branch_pool, 32, 1, (1, 1, 1))
        #Concatenate(axis=self._bn_axis, name=name + '_concat')([x, x1])
        x = Concatenate(

            axis=channel_axis,
            name='mixed0')([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # branch1x1 = self.conv3d_bn(x, 64, (1, 1, 1))
        #
        # branch5x5 = self.conv3d_bn(x, 48, (1, 1, 1))
        # branch5x5 = self.conv3d_bn(branch5x5, 64, (5, 5, 5))
        #
        # branch3x3dbl = self.conv3d_bn(x, 64, (1, 1, 1))
        # branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, (3, 3, 3))
        # branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, (3, 3, 3))
        #
        # branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        # branch_pool = self.conv3d_bn(branch_pool, 64, (1, 1, 1))
        # x = concatenate(
        #     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        #     axis=channel_axis,
        #     name='mixed1')
        #
        # # mixed 2: 35 x 35 x 256
        # branch1x1 = self.conv3d_bn(x, 64, (1, 1, 1))
        #
        # branch5x5 = self.conv3d_bn(x, 48, (1, 1, 1))
        # branch5x5 = self.conv3d_bn(branch5x5, 64, (5, 5, 5))
        #
        # branch3x3dbl = self.conv3d_bn(x, 64, (1, 1, 1))
        # branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, (3, 3, 3))
        # branch3x3dbl = self.conv3d_bn(branch3x3dbl, 96, (3, 3, 3))
        #
        # branch_pool = AveragePooling3D((3, 3, 3),
        #                                       strides=(1, 1, 1),
        #                                       padding='same')(x)
        # branch_pool = self.conv3d_bn(branch_pool, 64, (1, 1, 1))
        # x = concatenate(
        #     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        #     axis=channel_axis,
        #     name='mixed2')
        #
        # # mixed 3: 17 x 17 x 768
        # branch3x3 = conv3d_bn(x, 384, (3, 3, 3), strides=(2, 2, 2), padding='valid')
        #
        # branch3x3dbl = conv3d_bn(x, 64, (1, 1, 1))
        # branch3x3dbl = conv3d_bn(branch3x3dbl, 96, (3, 3, 3))
        # branch3x3dbl = conv3d_bn(
        #     branch3x3dbl, 96, (3, 3, 3), strides=(2, 2), padding='valid')
        #
        # branch_pool = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
        # x = concatenate(
        #     [branch3x3, branch3x3dbl, branch_pool],
        #     axis=channel_axis,
        #     name='mixed3')
        #
        # # mixed 4: 17 x 17 x 768
        # branch1x1 = conv3d_bn(x, 192, (1, 1, 1))
        #
        # branch7x7 = conv3d_bn(x, 128, (1, 1, 1))
        # branch7x7 = conv3d_bn(branch7x7, 128, 1, 7)
        # branch7x7 = conv3d_bn(branch7x7, 192, 7, 1)
        #
        # branch7x7dbl = conv3d_bn(x, 128, 1, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 7, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 1, 7)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 128, 7, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7)
        #
        # branch_pool = AveragePooling3D((3, 3),
        #                                       strides=(1, 1),
        #                                       padding='same')(x)
        # branch_pool = conv3d_bn(branch_pool, 192, 1, 1)
        # x = concatenate(
        #     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        #     axis=channel_axis,
        #     name='mixed4')
        #
        # # mixed 5, 6: 17 x 17 x 768
        # for i in range(2):
        #     branch1x1 = conv3d_bn(x, 192, 1, 1)
        #
        #     branch7x7 = conv3d_bn(x, 160, 1, 1)
        #     branch7x7 = conv3d_bn(branch7x7, 160, 1, 7)
        #     branch7x7 = conv3d_bn(branch7x7, 192, 7, 1)
        #
        #     branch7x7dbl = conv3d_bn(x, 160, 1, 1)
        #     branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 7, 1)
        #     branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 1, 7)
        #     branch7x7dbl = conv3d_bn(branch7x7dbl, 160, 7, 1)
        #     branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7)
        #
        #     branch_pool = AveragePooling3D(
        #         (3, 3), strides=(1, 1), padding='same')(x)
        #     branch_pool = conv3d_bn(branch_pool, 192, 1, 1)
        #     x = concatenate(
        #         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        #         axis=channel_axis,
        #         name='mixed' + str(5 + i))
        #
        # # mixed 7: 17 x 17 x 768
        # branch1x1 = conv3d_bn(x, 192, 1, 1)
        #
        # branch7x7 = conv3d_bn(x, 192, 1, 1)
        # branch7x7 = conv3d_bn(branch7x7, 192, 1, 7)
        # branch7x7 = conv3d_bn(branch7x7, 192, 7, 1)
        #
        # branch7x7dbl = conv3d_bn(x, 192, 1, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 7, 1)
        # branch7x7dbl = conv3d_bn(branch7x7dbl, 192, 1, 7)
        #
        # branch_pool = AveragePooling3DD((3, 3),
        #                                       strides=(1, 1),
        #                                       padding='same')(x)
        # branch_pool = conv3d_bn(branch_pool, 192, 1, 1)
        # x = concatenate(
        #     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        #     axis=channel_axis,
        #     name='mixed7')
        #
        # # mixed 8: 8 x 8 x 1280
        # branch3x3 = conv3d_bn(x, 192, 1, 1)
        # branch3x3 = conv3d_bn(branch3x3, 320, 3, 3,
        #                       strides=(2, 2), padding='valid')
        #
        # branch7x7x3 = conv3d_bn(x, 192, 1, 1)
        # branch7x7x3 = conv3d_bn(branch7x7x3, 192, 1, 7)
        # branch7x7x3 = conv3d_bn(branch7x7x3, 192, 7, 1)
        # branch7x7x3 = conv3d_bn(
        #     branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
        #
        # branch_pool = MaxPooling3D((3, 3), strides=(2, 2))(x)
        # x = layers.concatenate(
        #     [branch3x3, branch7x7x3, branch_pool],
        #     axis=channel_axis,
        #     name='mixed8')
        #
        # # mixed 9: 8 x 8 x 2048
        # for i in range(2):
        #     branch1x1 = conv3d_bn(x, 320, 1, 1)
        #
        #     branch3x3 = conv3d_bn(x, 384, 1, 1)
        #     branch3x3_1 = conv3d_bn(branch3x3, 384, 1, 3)
        #     branch3x3_2 = conv3d_bn(branch3x3, 384, 3, 1)
        #     branch3x3 = concatenate(
        #         [branch3x3_1, branch3x3_2],
        #         axis=channel_axis,
        #         name='mixed9_' + str(i))
        #
        #     branch3x3dbl = conv3d_bn(x, 448, 1, 1)
        #     branch3x3dbl = conv3d_bn(branch3x3dbl, 384, 3, 3)
        #     branch3x3dbl_1 = conv3d_bn(branch3x3dbl, 384, 1, 3)
        #     branch3x3dbl_2 = conv3d_bn(branch3x3dbl, 384, 3, 1)
        #     branch3x3dbl = concatenate(
        #         [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
        #
        #     branch_pool = AveragePooling3D(
        #         (3, 3), strides=(1, 1), padding='same')(x)
        #     branch_pool = conv3d_bn(branch_pool, 192, 1, 1)
        #     x = concatenate(
        #         [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        #         axis=channel_axis,
        #         name='mixed' + str(9 + i))

        if pooling == 'avg':
           x = GlobalAveragePooling3D(name='avg_pool')(x)
        elif pooling == 'max':
           x = GlobalMaxPooling3D(name='max_pool')(x)

        x = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(inputs=input, outputs=x)

        return model



if __name__ == "__main__":
    print("test")

    input_size = (97,79,68,1)
    inception3d = Inception3D(input_size)
    num_classes= 2
    inception3d_model = inception3d.inception3d(num_classes, pooling="avg")
    inception3d_model.summary()