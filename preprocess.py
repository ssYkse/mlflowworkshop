import numpy as np
from os.path import join as opj
import argparse
import os
import tempfile
import mlflow

parser = argparse.ArgumentParser(description='Preprocess all data.')
parser.add_argument('rawdir', type=str, 
                    help='Directory where the raw data is found.')
parser.add_argument('mean', type=float, 
                    help='Mean for normalization.')
parser.add_argument('std', type=float, 
                    help='Std for normalization.')

args = parser.parse_args()

tempdir = tempfile.mkdtemp()

#TODO: make preprocess function
def preprocess():
    x_train = x_train - args.mean
    x_train = x_train / args.std

if __name__ == '__main__':
    #Open data
    x_train = np.load(opj(args.rawdir, 'x_train.npy'))
    y_train = np.load(opj(args.rawdir, 'y_train.npy'))
    x_test = np.load(opj(args.rawdir, 'x_test.npy'))
    y_test = np.load(opj(args.rawdir, 'y_test.npy'))

    #Normalize data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_train = x_train - args.mean
    x_train = x_train / args.std

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
    x_test = x_test - args.mean
    x_test = x_test / args.std

    #Save data
    np.save(opj(tempdir, 'x_train'), x_train)
    np.save(opj(tempdir, 'y_train'), y_train)
    np.save(opj(tempdir, 'x_test'), x_test)
    np.save(opj(tempdir, 'y_test'), y_test)

    #Log to MLflow 
    mlflow.log_artifacts(tempdir)
    #TODO: Maybe remove below?
    mlflow.set_tag('mean', args.mean)
    mlflow.set_tag('std', args.std)

    print("Saved Preprocessed Data into ", mlflow.active_run())



