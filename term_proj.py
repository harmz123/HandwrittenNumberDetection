

# IMPORT ALL LIBRARIES

# import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import backend as k
from warnings import filterwarnings
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation, Dense
filterwarnings('ignore')
import matplotlib.pyplot as plt



# PREPARE TEST AND TRAIN DATASET FROM MNIST LIBRARY

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols=28, 28

if k.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    inpx = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    inpx = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train = tf.keras.utils.to_categorical(y_train)
y_test =  tf.keras.utils.to_categorical(y_test)


# CREATE MODELS AND PERFORM TRAINING
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=12, batch_size=500)

model2 = Sequential()
model2.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(img_rows, img_cols, 1)))
model2.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model2.add(TimeDistributed(Flatten()))
model2.add(LSTM(100))
model2.add(Dropout(0.5))
model2.add(Dense(100, activation='relu'))
model2.add(Dense(10, activation='softmax'))

# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=12, batch_size=500)

# SHOW MODEL SUMMARY
print(model.summary())
print(model2.summary())


# SHOW LOSS AND ACCURACY AFTER TESTING
score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('accuracy=', score[1])

score2 = model2.evaluate(x_test, y_test, verbose=0)
print('loss=', score2[0])
print('accuracy=', score2[1])



# VISUALIZE RESULTS



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['val_accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('testing accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['cnn', 'cnn-lstm'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['val_loss'])
plt.plot(history2.history['val_loss'])
plt.title('testing loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['cnn', 'cnn-lstm'], loc='upper left')
plt.show()


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history2.history['accuracy'])
plt.title('training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['cnn', 'cnn-lstm'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history2.history['loss'])
plt.title('training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['cnn', 'cnn-lstm'], loc='upper left')
plt.show()