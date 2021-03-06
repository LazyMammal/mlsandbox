from __future__ import print_function
import collections
import numpy as np
import keras
from keras.applications.mobilenet import MobileNet
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, GlobalAveragePooling2D
from keras.utils import layer_utils
from keras import backend as K
from keras.backend import tf as ktf
from keras.callbacks import EarlyStopping


def flatten(l):
    if isinstance(l, collections.Iterable) and not isinstance(l, basestring):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
    else:
        yield l


def weights_summary(weights):
    #weights_count = int(np.sum([K.count_params(p) for p in set(weights)]))
    weights_values = list(flatten(K.batch_get_value(weights)))
    weights_count = len(weights_values)
    weights_mean = np.mean(weights_values) if weights_values else np.nan
    return weights_mean, weights_count


def print_trainable_summary(model):
    trainable_mean, trainable_count = weights_summary(model.trainable_weights)
    non_trainable_mean, non_trainable_count = weights_summary(model.non_trainable_weights)

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}, {:,}'.format(trainable_count, trainable_mean))
    print('Non-Trainable params: {:,}, {:,}'.format(non_trainable_count, non_trainable_mean))


def load_data():
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = np.repeat(x_train, 3, axis=3)
    x_test = np.repeat(x_test, 3, axis=3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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


def print_model_summary(models):
    for name, model in models:
        print(name)
        print_trainable_summary(model)


def train(model, (x_train, y_train), (x_test, y_test)):
    before_weights = {}
    for layer in model.layers:
        weights = layer.get_weights()
        before_weights[layer.name] = [np.array(list(flatten(p))) for p in weights]

    batch_size = 256
    epochs = 1
    print("training one batch ...")
    model.train_on_batch(x_train[:batch_size], y_train[:batch_size])
    '''
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    '''
    '''
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''

    print("compare weights before and after training")
    for layer in model.layers:
        print(layer.name, type(layer), isinstance(layer, keras.layers.normalization.BatchNormalization))
        weights = layer.get_weights()
        for a, b in zip([np.array(list(flatten(p))) for p in weights], before_weights[layer.name]):
            if np.array_equiv(a, b):
                print(" array_equiv")
            else:
                print(np.histogram(a - b))


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = get_model(input_shape=x_train[0].shape, num_classes=y_train.shape[1])
    train(model, (x_train, y_train), (x_test, y_test))


if __name__ == '__main__':
    main()
