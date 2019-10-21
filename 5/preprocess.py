import numpy as np
from os.path import join as opj
import os
import tempfile
import mlflow



def preprocess(x_train, mean, std):
    #Normalize data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_train = x_train - mean
    x_train = x_train / std

    return x_train


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess all data.')
    parser.add_argument('rawdir', type=str, 
                        help='Directory where the raw data is found.')
    parser.add_argument('percentage', type=int, 
                        help='percentage of data to remove')
    parser.add_argument('mean', type=float, 
                        help='Mean for normalization.')
    parser.add_argument('std', type=float, 
                        help='Std for normalization.')
    args = parser.parse_args()
    percentage = args.percentage


    tempdir = tempfile.mkdtemp()

    #Open data
    x_train = np.load(opj(args.rawdir, 'x_train.npy'))[:int(percentage/100*60000)]
    y_train = np.load(opj(args.rawdir, 'y_train.npy'))[:int(percentage/100*60000)]
    x_test = np.load(opj(args.rawdir, 'x_test.npy'))[:int(percentage/100*10000)]
    y_test = np.load(opj(args.rawdir, 'y_test.npy'))[:int(percentage/100*10000)]


    #Do the preprocessing
    x_train= preprocess(x_train, args.mean, args.std)
    x_test = preprocess(x_test , args.mean, args.std)

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



