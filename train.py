from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from os.path import join as opj
import mlflow
import mlflow.keras

import argparse

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('batch_size', type=int, 
                    help='Batch size in training.')
parser.add_argument('epochs', type=int,
                    help='Nr. of epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Learning rate for training.')
parser.add_argument('input_dir', type=str,
                    help='artifact dir of a preprocessing run.')
args = parser.parse_args()

# load images
x_train = np.load(opj(args.input_dir, 'x_train.npy'))
y_train = np.load(opj(args.input_dir, 'y_train.npy'))
x_test = np.load(opj(args.input_dir, 'x_test.npy'))
y_test = np.load(opj(args.input_dir, 'y_test.npy'))


x_train = x_train[:6000]
y_train = y_train[:6000]
x_test = x_test[:100]
y_test = y_test[:100]

# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
num_classes = 10
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4),
                       strides=4))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(learning_rate=args.lr),
              metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate model
score = model.evaluate(x_test, y_test, verbose=0)

mlflow.set_tag('input_data', args.input_dir)
mlflow.log_metric('loss', score[0])
mlflow.log_metric('accuracy', score[1])
mlflow.keras.log_model(model, "myModel")