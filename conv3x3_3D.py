import os

from keras import backend as K
from keras.backend import categorical_crossentropy
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense, Concatenate, Lambda
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.activations import softmax

class CNN3D():
    def __init__(self, numClasses):
        self._numClasses = numClasses

    def cnn3d(self, input_shape, num_classes=2) :

        # Define model
        model = Sequential()

        model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape= input_shape, border_mode='same'))

        model.add(Activation('relu'))
        model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        #model.add(GlobalAveragePooling3D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # model.compile(loss=categorical_crossentropy,
        #               optimizer=Adam(), metrics=['accuracy'])
        # model.summary()
        return model