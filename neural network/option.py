import tensorflow as tf
import keras
from PIL import Image
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np


(X_train, y_train), (X_test, y_test) = mnist.load_data()

new_data = []
for i in range(457):
    im = Image.open(f'data/{i}.jpeg')
    im = im.convert('L')
    ar = np.asarray(im)
    new_data.append(ar)
new_data = new_data * 5
new_data_train = new_data[:round(0.8*len(new_data))]
X_train = np.concatenate((X_train, new_data_train))

new_y = [10] * len(new_data_train)
y_train = np.append(y_train, new_y)

new_data_test = new_data[round(0.8*len(new_data)):]
X_test = np.concatenate((X_test, new_data_test))

new_y_test = [10] * len(new_data_test)
y_test = np.append(y_test, new_y_test)

X_train = X_train/255
X_test = X_test/255

y_train = keras.utils.to_categorical(y_train, 11)
y_test = keras.utils.to_categorical(y_test, 11)

model = Sequential()

print(X_train)
print(X_train[0])
print(X_train[0].shape)

model.add(Dense(32, activation='relu', input_shape=(X_train[0].shape)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Flatten())

model.add(Dense(11, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)

model.evaluate(X_test, y_test)

model.save('option.h5')
