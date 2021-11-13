# Identity Mappings in Deep Residual Networks (He et al., 2016)


# https://github.com/pantheon5100/3D-CNN-resnet-keras/blob/master/Cre_Model.py

# reference for resnet structure
# https://m.blog.naver.com/laonple/220793640991

#  Moddel_2_118 Design
# Add Dropout!!!
from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense, Concatenate, Lambda
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model
from keras.activations import softmax

class Resnet3D():
    def __init__(self, numClasses):
        self._numClasses = numClasses

    # full pre-activation
    def Conv3d_BN(self, x, nb_filter, kernel_size, strides=1, padding='same', name=None):
        print("debug", x, x.shape.as_list())
        x = BatchNormalization(name=name+"_bn")(x)
        x = Activation('relu', name=name+"_relu")(x)
        x = Conv3D(nb_filter, kernel_size, padding=padding, data_format='channels_last', strides=strides, name=name+"_conv")(x)

        return x

    def residual_block(self, input, nb_filter, kernel_size, strides=1, with_conv_shortcut=False, DropoutRate=0.2, name=None, addChannel=None):

        print("residual_block input", input)
        x = self.Conv3d_BN(input, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same', name=name+"_BN1")
        print("residual_block 1", x)
        x = self.Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same', name=name+"_BN2")
        print("residual_block 2", x)

        if with_conv_shortcut:
            shortcut = self.Conv3d_BN(input, nb_filter=nb_filter, strides=strides,
                                 kernel_size=kernel_size, name=name+"_BNSC")
            x = Dropout(DropoutRate)(x)
            if strides == 2:
                shortcut = MaxPooling3D(strides=(2, 2, 2), name=name+"_BNSCP")(shortcut)
            x = add([x, shortcut])
            return x
        else:
            if strides == 2:
                input = MaxPooling3D(strides=(2, 2, 2), name=name+"_BNSCP")(input)
                #print("input_shape", K.int_shape(input)) # input.shape.as_list()
                #_, _D, _H, _W, _C = input.shape.as_list() # [None, 32, 32, 32, 64]
                #input_pad = K.zeros(shape=(_D, _H, _W, nb_filter-_C))
                # input_pad = K.zeros_like(input)
                #input = Concatenate([input, input_pad])
                # input = K.concatenate([input, input_pad])
                #input = ZeroPadding3D((0, 0, 0), data_format="channels_last")(input)
                #
                # def padding_channel(input_tensor):
                #     _, _D, _H, _W, _C = input_tensor.shape.as_list()
                #     #input_pad = K.zeros(shape=(_D, _H, _W, nb_filter - _C))
                #     input_pad = K.zeros_like(input_tensor)
                #     input = K.concatenate([input_tensor, input_pad])
                #     return input
                # input = Lambda(padding_channel, output_shape=x.shape.as_list())(input)
                # print("strides 2", input)
        if addChannel == "conv":
            input = Conv3D(nb_filter, (1, 1, 1), padding='same', data_format='channels_last', strides=(1, 1, 1),
                       name=name + "_conv")(input)
        x = add([x, input], name=name+"_add")
        return x

    # def bottlneck_Block(inpt, nb_filter, strides=1, with_conv_shortcut=False):
    #     k1, k2, k3 = nb_filter
    #     x = Conv3d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    #     x = Conv3d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    #     x = Conv3d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    #     if with_conv_shortcut:
    #         shortcut = Conv3D(inpt, nb_filter=k3, data_format='channels_first', strides=strides, kernel_size=1)
    #         x = add([x, shortcut])
    #         return x
    #     else:
    #         x = add([x, inpt])
    #         return x

    def resnet(self, input_shape, DropoutRate=0.2, zero_pad_1st=False, feature_output=False):
        input = Input(shape=input_shape)
        if zero_pad_1st :
            x = ZeroPadding3D((1, 1, 1), data_format='channels_last')(input)
            # conv1
            x = self.residual_block(x, nb_filter=32, kernel_size=(3, 3, 3), strides=1, with_conv_shortcut=False, name="resBlock1_1")
        else :
            # conv1
            x = self.residual_block(input, nb_filter=32, kernel_size=(3, 3, 3), strides=2, with_conv_shortcut=False, name="resBlock1_1")
        x = self.residual_block(x, nb_filter=32, kernel_size=(3, 3, 3), name="resBlock1_2")
        x = self.residual_block(x, nb_filter=32, kernel_size=(3, 3, 3), name="resBlock1_3")

        # conv2
        x = self.residual_block(x, nb_filter=64, kernel_size=(3, 3, 3), strides=2, with_conv_shortcut=False, name="resBlock2_1", addChannel="conv")
        x = self.residual_block(x, nb_filter=64, kernel_size=(3, 3, 3), strides=1, name="resBlock2_2")
        x = self.residual_block(x, nb_filter=64, kernel_size=(3, 3, 3), name="resBlock2_3")
        x = self.residual_block(x, nb_filter=64, kernel_size=(3, 3, 3), name="resBlock2_4")

        # conv3
        x = self.residual_block(x, nb_filter=128, kernel_size=(3, 3, 3), strides=2, with_conv_shortcut=False,
                                name="resBlock3_1", addChannel="conv")
        x = self.residual_block(x, nb_filter=128, kernel_size=(3, 3, 3), strides=1, name="resBlock3_2")
        # x = self.residual_block(x, nb_filter=128, kernel_size=(3, 3, 3), name="resBlock3_3")
        # x = self.residual_block(x, nb_filter=128, kernel_size=(3, 3, 3), name="resBlock3_4")

        # conv4
        x = self.residual_block(x, nb_filter=256, kernel_size=(3, 3, 3), strides=1, with_conv_shortcut=False,
                                name="resBlock4_1", addChannel="conv")
        x = self.residual_block(x, nb_filter=256, kernel_size=(3, 3, 3), strides=1, name="resBlock4_2")
        # x = self.residual_block(x, nb_filter=256, kernel_size=(3, 3, 3), name="resBlock4_3")
        # x = self.residual_block(x, nb_filter=256, kernel_size=(3, 3, 3), name="resBlock4_4")

        gap = GlobalAveragePooling3D(name="GAP")(x)
        print("gap", gap)

        x = BatchNormalization()(gap)
        x = Activation('relu', name="GAP_activation")(x)
#        x = Dropout(rate=0.4, name="GAP_activation_dropout")(x)
        print("Dropout", x)

        # x = Flatten()(x)
        # print("Flatten", x)

        last_dense = Dense(self._numClasses, use_bias = False, name="last_dense")(x)
        y = Activation('softmax', name="last_softmax")(last_dense)
        #y = softmax()(last_dense)
        print("dense", y)
        if feature_output :
            model = Model(inputs=input, outputs=gap)
        else :
            model = Model(inputs=input, outputs=y)

        return model

if __name__ == "__main__":
    print("test")
    import keras

    input1 = keras.layers.Input(shape=(16,))
    x1 = keras.layers.Dense(8, activation='relu')(input1)
    input2 = keras.layers.Input(shape=(32,))
    x2 = keras.layers.Dense(8, activation='relu')(input2)
    added = keras.layers.add([x1, x2])

    out = keras.layers.Dense(4)(added)
    model = keras.models.Model(inputs=[input1, input2], outputs=out)
    model.summary()