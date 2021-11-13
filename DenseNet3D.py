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


class DenseNet3D():
    def __init__(self, blocks, input_size):

        self._blocks = blocks # list of growth_rate
        self._input_size = input_size
        self._bn_axis = 4 if K.image_data_format() == 'channels_last' else 1

    def densenet(self, num_classes, init_num_filters=64, init_filter_size=7, pooling="avg"):

        input = Input(shape=self._input_size)

        x = ZeroPadding3D(padding=((3, 3), (3, 3), (3, 3)))(input)
        x = Conv3D(init_num_filters, init_filter_size, strides=2, use_bias=False, name='conv1/conv')(x)
        x = BatchNormalization(
            axis=self._bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = Activation('relu', name='conv1/relu')(x)
        x = ZeroPadding3D(padding=((3, 3), (3, 3), (3, 3)))(x)
        x = MaxPooling3D(3, strides=2, name='pool1')(x)

        x = self.dense_block(x, self._blocks[0], name='conv2')
        x = self.transition_block(x, 0.5, name='pool2')

        x = self.dense_block(x, self._blocks[1], name='conv3')
        #x = self.transition_block(x, 0.5, name='pool3')
        x = self.dense_block(x, self._blocks[2], name='conv4')
        #x = self.transition_block(x, 0.5, name='pool4')
        x = self.dense_block(x, self._blocks[3], name='conv5')

        x = BatchNormalization(
            axis=self._bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = Activation('relu', name='relu')(x)

        if pooling == 'avg':
            x = GlobalAveragePooling3D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D(name='max_pool')(x)

        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        print("x", x)
        model = Model(inputs=input, outputs=x)
        print("model", model)
        return model



    def dense_block(self, x, num_conv_block, growth_rate=32, name=None):
        """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(num_conv_block):
            x = self.conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
        return x

    def transition_block(self, x, reduction, name):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        self._bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        x = BatchNormalization(axis=self._bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv3D(int(K.int_shape(x)[self._bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv')(x)
        x = AveragePooling3D(2, strides=2, name=name + '_pool')(x)
        return x

    def conv_block(self, x, growth_rate, name=None):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """
        self._bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
        x1 = BatchNormalization(axis=self._bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv3D(4 * growth_rate , 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=self._bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv3D(growth_rate , 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x1)
        x = Concatenate(axis=self._bn_axis, name=name + '_concat')([x, x1])
        return x

    # full pre-activation
    # def Conv3d_BN(self, x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    #     print("debug", x, x.shape.as_list())
    #     x = BatchNormalization(name=name+"_bn")(x)
    #     x = Activation('relu', name=name+"_relu")(x)
    #     x = Conv3D(nb_filter, kernel_size, padding=padding, data_format='channels_last', strides=strides, name=name+"_conv")(x)
    #     return x
    #
    # # def conv_block(self, input, nb_filter, growth_rate = 6, kernel_size, strides=1, bottleneck = False, with_conv_shortcut=False, DropoutRate=0.2, name=None, addChannel=None):
    # #    if bottleneck :
    # #         inter_channel = nb_filter * growth_rate
    # #         x = Conv3D(inter_channel, (1, 1, 1), padding=padding, strides=strides)(x)
    # #         x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    # #         x = Activation('relu')(x)
    # #     x = Conv3D(nb_filter, (3, 3, 3), padding=padding, strides=strides)(x)
    # #     return x
    #
    # def dense_block(self, input, nb_filter, growth_rate, kernel_size, strides=1, bottleneck = False, with_conv_shortcut=False, DropoutRate=0.2, name=None, addChannel=None):
    #
    #     x_list = [x]
    #
    #     for i in range(nb_layers):
    #         cb = conv_block(x, growth_rate, bottleneck, DropoutRate)
    #         x_list.append(cb)
    #
    #         x = concatenate([x, cb], axis=concat_axis)
    #
    #         if grow_nb_filters:
    #             nb_filter += growth_rate
    #
    #     if return_concat_list:
    #         return x, nb_filter, x_list
    #     else:
    #         return x, nb_filter
    #     return x
    #
    # # def bottlneck_Block(inpt, nb_filter, strides=1, with_conv_shortcut=False):
    # #     k1, k2, k3 = nb_filter
    # #     x = Conv3d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    # #     x = Conv3d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    # #     x = Conv3d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    # #     if with_conv_shortcut:
    # #         shortcut = Conv3D(inpt, nb_filter=k3, data_format='channels_first', strides=strides, kernel_size=1)
    # #         x = add([x, shortcut])
    # #         return x
    # #     else:
    # #         x = add([x, inpt])
    # #         return x




if __name__ == "__main__":
    print("test")
    import keras

    # input1 = keras.layers.Input(shape=(16,))
    # x1 = keras.layers.Dense(8, activation='relu')(input1)
    # input2 = keras.layers.Input(shape=(32,))
    # x2 = keras.layers.Dense(8, activation='relu')(input2)
    # added = keras.layers.add([x1, x2])
    #
    # out = keras.layers.Dense(4)(added)
    # model = keras.models.Model(inputs=[input1, input2], outputs=out)
    # model.summary()

    from keras.models import Model, Sequential
    from keras.layers import *

    from keras.applications.densenet import DenseNet201


    blocks =[6, 12, 24, 16] # [6, 12, 4, 4]
    input_size = (97,79,68, 1)
    densenet3d = DenseNet3D(blocks, input_size)
    num_classes= 2
    densenet3d_model = densenet3d.densenet(num_classes, init_num_filters=64, init_filter_size=7, pooling="avg")
    # densenet3d_model.summary()

# x = model.output
    #
    # transfer_layer = model.get_layer('block5_pool')
    # conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
    #
    # for layer in conv_model.layers:
    #     layer.trainable = True
    #
    # new_model = Sequential()
    # new_model.add(conv_model)
    # new_model.add(GlobalAveragePooling2D())

