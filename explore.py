import numpy as np
from os.path import join as opj
import argparse

parser = argparse.ArgumentParser(description='Download either MNIST numbers or fashion')
parser.add_argument('outdir', type=str, 
                    help='Directory where to save the raw data.')
args = parser.parse_args()

# Load Data
x_train = np.load(opj(args.outdir, 'x_train.npy'))
y_train = np.load(opj(args.outdir, 'y_train.npy'))
x_test = np.load(opj(args.outdir, 'x_test.npy'))
y_test = np.load(opj(args.outdir, 'y_test.npy'))

#Print Interesting info:
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

print('Train', x_train.min(), x_train.max(), x_train.mean(), x_train.std())
print('Test', x_test.min(), x_test.max(), x_test.mean(), x_test.std())




