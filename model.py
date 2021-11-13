import os

from keras.models import Model, Sequential
from keras.layers import *
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetMobile
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Embedding, ZeroPadding2D
from Resnet3D import Resnet3D
from convnets_keras import AlexNet
from conv3x3_3D import CNN3D
from DenseNet3D import DenseNet3D
from Inception3DModel import Inception3D

def load_simple_DNN(learning_rate, num_classes=10, input_shape=(28, 28, 1)):

    inputs = create_model_inputs()
    features = encode_inputs(inputs)

    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    return model

def create_alexnet_model(learning_rate, numClasses, input_shape, weights_path=None, heatmap=False):
    alex_model = AlexNet(weights_path=weights_path, heatmap=heatmap)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_3d_cnn_model(learning_rate, numClasses, input_shape):
    res3d_model = Resnet3D(numClasses=numClasses)
    # model_name = res3d_model.resnet(input_shape=(64, 64, 64, 1), DropoutRate=0.2, zero_pad_1st=False)
    model = res3d_model.resnet(input_shape=input_shape, DropoutRate=0.2, zero_pad_1st=False)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_3d_conv_model(learning_rate, numClasses, input_shape):
    conv3D_model = CNN3D(numClasses)
    model = conv3D_model.cnn3d(input_shape, numClasses)
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_3d_densenet_model(learning_rate, numClasses, input_shape):
    densenet3D_model = DenseNet3D([6, 12, 24, 16], input_shape)
    model = densenet3D_model.densenet(numClasses)
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_PDNet_model(learning_rate, numClasses, input_shape):
    new_model = Sequential()

    new_model.add(Conv3D(16, kernel_size=(3, 3, 3), strides=2, padding="valid", input_shape=input_shape))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling3D(pool_size=3, strides=2))

    new_model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=2, padding="valid"))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling3D(pool_size=3, strides=2))

    new_model.add(Conv3D(256, kernel_size=3, strides=1, padding="valid"))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))

    new_model.add(Flatten())
    new_model.add(Dense(numClasses, activation='softmax'))

    #input = Input(shape=(91, 109, 91, 1))
    #model_output = new_model(input)

    model = Model(inputs=new_model.input, outputs=new_model.output)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    return model

def create_3d_inception_model(learning_rate, numClasses, input_shape):
    inception3D_model = Inception3D(input_shape)
    model = inception3D_model.inception3d(numClasses)
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model(learning_rate, num_dense_layers, num_dense_nodes, num_classes=3):
    model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable=False
    transfer_layer = model.get_layer('block5_pool')

    conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
    for layer in conv_model.layers:
        layer.trainable = False
    new_model = Sequential()
    new_model.add(conv_model)
    new_model.add(Flatten())
    for i in range(int(num_dense_layers)):
        name = 'layer_dense_{0}'.format(i + 1)
        new_model.add(Dense(int(num_dense_nodes), activation='relu', name=name))
    new_model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model


def create_model_with_label_cond(learning_rate, num_dense_layers, num_dense_nodes, num_classes=3):
    img_shape = (224, 224, 3)
    model = VGG16(include_top=False, weights='imagenet', input_shape=img_shape)
    transfer_layer = model.get_layer('block5_pool')

    conv_model = Model(inputs=model.input, outputs=transfer_layer.output)
    for layer in conv_model.layers:
        layer.trainable = False
    new_model = Sequential()
    new_model.add(conv_model)
    new_model.add(Flatten())
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        new_model.add(Dense(num_dense_nodes, activation='relu', name=name))
    new_model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)

    img = Input(shape = img_shape)
    label = Input(shape=(1,), dtype='int32')
    # embe label portion into the flatted size of img shape
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)
    model_input = multiply([flat_img, label_embedding])
    model_input = Reshape(img_shape)(model_input)


    # print("debug1", flat_img) # Tensor("flatten_3/Reshape:0", shape=(?, ?), dtype=float32)
    # print("debug1", label_embedding) # Tensor("flatten_2/Reshape:0", shape=(?, ?), dtype=float32)

    output = new_model(model_input)
    model = Model([img, label], output)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_densenet_model(learning_rate, num_classes=10, input_shape=(224, 224, 3)):
    densenet_model = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in densenet_model.layers:
        if "bn" in layer.name:
            continue
        layer.trainable=False

    x = densenet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=densenet_model.input, outputs=x)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet_model(learning_rate, num_classes=10, input_shape=(28, 28, 3)):
    res_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in res_model.layers:
        if "bn" in layer.name:
            continue
        layer.trainable=False

    x = res_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=res_model.input, outputs=x)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_vgg16_model(learning_rate, num_classes=10, input_shape=(28, 28, 1)):
    vgg16_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in vgg16_model.layers:
        layer.trainable=False

    x = vgg16_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(vgg16_model.input, x)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def print_layer_trainable(model):
    for layer in model.layers:
        print("{0}: {1}".format(layer.trainable, layer.name))

def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def create_inception_v3_model(learning_rate, num_classes=10, input_shape=(139, 139, 3)):
    print("[!] create Inception v3")
    inception_v3_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in inception_v3_model.layers:
        if "batch_normalization_" in layer.name:
            continue
        layer.trainable=False
    x = inception_v3_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inception_v3_model.input, x)

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# remove extention portion from whole filename
def get_only_filename(file_path):
    basename = os.path.basename(file_path)
    return basename.split(".")[0]

# save model_name into json, the model_name parameter into *.h5
def save_model(model, model_path):
    model_json = model.to_json()
    print("debug", model_json)
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    p_dir = os.path.dirname(model_path)
    only_name = get_only_filename(model_path)
    model_param_save_name = only_name+".h5"
    model.save_weights(os.path.join(p_dir, model_param_save_name))
    return

# # load model_name from json, the model_name parameter from *.h5
# def load_model(model_path):
#     with open(model_path, "r") as json_file:
#         loaded_model_json = json_file.read()
#         print("loaded_model_json", loaded_model_json)
#     loaded_model = model_from_json(loaded_model_json)
#
#     p_dir = os.path.dirname(model_path)
#     only_name = get_only_filename(model_path)
#     loaded_model.load_weights(os.path.join(p_dir, only_name+".h5"))
#     return loaded_model

if __name__ == "__main__":
    lr_rate = 0.0005
    num_classes = 2
    #model_name = create_vgg16_model(learning_rate=lr_rate, num_classes=num_classes, input_shape=(48, 48, 3)) # 14,715,714 -> 512*3 Dense : 23,608,202
    #model_name = create_inception_v3_model(learning_rate=lr_rate, num_classes=num_classes, input_shape=(139, 139, 3)) # 21,800,000
    #model_name = create_resnet_model(lr_rate, num_classes=2, input_shape=(197, 197, 3)) # 23,608,202
    # model_name = create_densenet_model(lr_rate, num_classes=10, input_shape=(197, 197, 3)) # 7,047,754
    #vgg_model = create_model(lr_rate, num_dense_layers=3, num_dense_nodes=128, num_classes=2)
    #model_name = create_3d_cnn_model(lr_rate, num_classes)
    model = create_PDNet_model(lr_rate, numClasses=2, input_shape=(64, 64, 64, 1))
    model.summary()
    #print_layer_trainable(vgg_model)

    #model1= model_name# model_name.summary()
    #print_layer_trainable(model_name)
