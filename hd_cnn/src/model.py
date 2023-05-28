from keras.layers import (
    Input,
    Conv2D,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense,
    Rescaling,
    BatchNormalization,
    GaussianNoise,
    MaxPool2D,
)
from keras.models import Model
from keras import optimizers
from keras.regularizers import L2
import tensorflow as tf

from .config import *


def single_classifier_model(
    learning_rate, img_width, img_height, img_channel, class_num
):
    in_layer = Input(
        shape=(img_width, img_height, img_channel), dtype="float32", name="main_input"
    )
    scaled_input = Rescaling(1.0 / 255)(in_layer)

    net = Conv2D(384, 3, strides=1, padding="same", activation="elu")(scaled_input)
    net = MaxPooling2D((2, 2), padding="valid")(net)

    net = Conv2D(384, 1, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(384, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(640, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(640, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.2)(net)
    net = MaxPooling2D((2, 2), padding="valid")(net)

    net = Conv2D(640, 1, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(768, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(768, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(768, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.3)(net)
    net = MaxPooling2D((2, 2), padding="valid")(net)

    net = Conv2D(768, 1, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(896, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(896, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.4)(net)
    net = MaxPooling2D((2, 2), padding="valid")(net)

    net = Conv2D(896, 3, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(1024, 2, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(1024, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.5)(net)
    net = MaxPooling2D((2, 2), padding="valid")(net)

    net = Conv2D(1024, 1, strides=1, padding="same", activation="elu")(net)
    net = Conv2D(1152, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.6)(net)
    net = MaxPooling2D((2, 2), padding="same")(net)

    net = Flatten()(net)
    net = Dense(1152, activation="elu")(net)
    output = Dense(class_num, activation="softmax")(net)

    model = Model(inputs=in_layer, outputs=output)
    sgd_optimizers = optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )
    model.compile(
        optimizer=sgd_optimizers,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


def coarse_classifier_model(learning_rate, load_weight=False):
    model = single_classifier_model(
        SINGLE_CLASSIFIER_MODEL_LEARNING_RATE,
        HD_CNN_IMG_WIDTH,
        HD_CNN_IMG_HEIGHT,
        HD_CNN_IMG_CHANNEL,
        FINE_CLASS_NUM,
    )
    model.load_weights(SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH)
    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    net = Conv2D(1024, 1, strides=1, padding="same", activation="elu")(
        model.layers[-8].output
    )
    net = Conv2D(1152, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.6)(net)
    net = MaxPooling2D((2, 2), padding="same")(net)

    net = Flatten()(net)
    net = Dense(1152, activation="elu")(net)
    output = Dense(COARSE_CLASS_NUM, activation="softmax")(net)

    coarse_model = Model(inputs=model.input, outputs=output)
    sgd_optimizers = optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )
    coarse_model.compile(
        optimizer=sgd_optimizers,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    for i in range(len(coarse_model.layers) - 1):
        coarse_model.layers[i].set_weights(model.layers[i].get_weights())

    if load_weight:
        coarse_model.load_weights(COARSE_CLASSIFIER_MODEL_WEIGHTS_PATH)

    return coarse_model


def fine_classifier_model(learning_rate, load_weight=False, class_idx=-1):
    model = single_classifier_model(
        SINGLE_CLASSIFIER_MODEL_LEARNING_RATE,
        HD_CNN_IMG_WIDTH,
        HD_CNN_IMG_HEIGHT,
        HD_CNN_IMG_CHANNEL,
        FINE_CLASS_NUM,
    )
    # model.load_weights(SINGLE_CLASSIFIER_MODEL_WEIGHTS_PATH)
    # for i in range(len(model.layers)):
    #     model.layers[i].trainable = False

    net = Conv2D(1024, 1, strides=1, padding="same", activation="elu")(
        model.layers[-8].output
    )
    net = Conv2D(1152, 2, strides=1, padding="same", activation="elu")(net)
    net = Dropout(0.6)(net)
    net = MaxPooling2D((2, 2), padding="same")(net)

    net = Flatten()(net)
    net = Dense(1152, activation="elu")(net)
    output = Dense(FINE_CLASS_NUM, activation="softmax")(net)

    fine_model = Model(inputs=model.input, outputs=output)
    sgd_optimizers = optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )
    fine_model.compile(
        optimizer=sgd_optimizers,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # for i in range(len(fine_model.layers) - 1):
    #     fine_model.layers[i].set_weights(model.layers[i].get_weights())

    if load_weight:
        fine_model.load_weights(
            FINE_CLASSIFIER_MODEL_WEIGHTS_PATH.format(str(class_idx))
        )

    return fine_model


def vgg16_model(learning_rate):
    # input
    input = Input(shape=(HD_CNN_IMG_WIDTH, HD_CNN_IMG_HEIGHT, HD_CNN_IMG_CHANNEL))

    base_model = tf.keras.applications.vgg16.VGG16(
        weights=None, input_tensor=input, include_top=False
    )

    # Fully connected layers
    flatten = Flatten()(base_model.output)
    fc = Dense(
        units=4096,
        activation="relu",
    )(flatten)
    fc = Dense(
        units=4096,
        activation="relu",
    )(fc)
    output = Dense(FINE_CLASS_NUM, activation="softmax")(fc)

    # creating the model
    model = Model(inputs=input, outputs=output)
    optimizer = optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model
