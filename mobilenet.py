from __future__ import print_function
import keras
from keras.applications.mobilenet import MobileNet
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, GlobalAveragePooling2D
import numpy as np
from keras.backend import tf as ktf


batch_size = 128
num_classes = 10
epochs = 12
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
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (img_rows, img_cols, 3)
inp = Input(shape=input_shape)
resize = Lambda(lambda image: ktf.image.resize_images(image, (128, 128)))(inp)
mobnet = MobileNet(input_shape=(128, 128, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3,
                   include_top=False, weights='imagenet', input_tensor=resize)
mobnet.train = False
x = mobnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(num_classes, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=mobnet.input, outputs=predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
