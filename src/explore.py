import numpy as np
from os.path import join as opj
import argparse
import mlflow

parser = argparse.ArgumentParser(description='Download either MNIST numbers or fashion')
parser.add_argument('rawdir', type=str, 
                    help='Directory where the raw data is found')
args = parser.parse_args()

# Load Data
x_train = np.load(opj(args.rawdir, 'x_train.npy'))
y_train = np.load(opj(args.rawdir, 'y_train.npy'))
x_test = np.load(opj(args.rawdir, 'x_test.npy'))
y_test = np.load(opj(args.rawdir, 'y_test.npy'))

#Print Interesting info:
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

print('Train', x_train.min(), x_train.max(), x_train.mean(), x_train.std())
print('Test', x_test.min(), x_test.max(), x_test.mean(), x_test.std())

mlflow.log_metric("mean", x_train.mean())
mlflow.log_metric("std", x_train.std())



