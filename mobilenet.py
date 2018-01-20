from __future__ import print_function
import numpy as np
import keras
from keras.applications.mobilenet import MobileNet
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D
from keras.backend import tf as ktf


def load_data():
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = np.repeat(x_train, 3, axis=3)
    x_test = np.repeat(x_test, 3, axis=3)

    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def get_model(input_shape, num_classes):
    inp = Input(shape=input_shape)
    resize = Lambda(lambda image: ktf.image.resize_images(image, (128, 128)))(inp)
    mobnet = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3,
                       include_top=False, weights='imagenet', input_tensor=resize)
    for layer in mobnet.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            layer._per_input_updates = {}
    mobnet_output = mobnet.output

    x = GlobalAveragePooling2D()(mobnet_output)
    x = Dense(num_classes, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=mobnet.input, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train(model, (x_train, y_train), (x_test, y_test)):
    batch_size = 256
    epochs = 1
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = get_model(input_shape=x_train[0].shape, num_classes=y_train.shape[1])
    train(model, (x_train, y_train), (x_test, y_test))


if __name__ == '__main__':
    main()
