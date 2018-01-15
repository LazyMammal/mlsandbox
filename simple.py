from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Reshape, GlobalAveragePooling1D
from keras import backend as K


data = np.random.random((1000, 10))
labels = np.random.randint(2, size=(1000, 1))

model = Sequential()
model.add(Dense(9, activation='relu', input_dim=10))
model.add(Reshape((3,3)))
model.add(Conv1D(4, kernel_size=(2)))
model.add(GlobalAveragePooling1D())
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
lnum = 2

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.layers[lnum].get_weights()[0])
print(model.layers[lnum].get_weights()[1])

model.fit(data, labels)
print(model.layers[lnum].get_weights()[0])
print(model.layers[lnum].get_weights()[1])

model.layers[lnum].trainable = False
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels)
print(model.layers[lnum].get_weights()[0])
print(model.layers[lnum].get_weights()[1])
