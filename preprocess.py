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
    parser.add_argument('mean', type=float, 
                        help='Mean for normalization.')
    parser.add_argument('std', type=float, 
                        help='Std for normalization.')
    parser.add_argument('outdir', type=str, 
                        help='Directory where the raw preprocessed data is to be stored.')
    args = parser.parse_args()

    #Open data
    x_train = np.load(opj(args.rawdir, 'x_train.npy'))
    y_train = np.load(opj(args.rawdir, 'y_train.npy'))
    x_test = np.load(opj(args.rawdir, 'x_test.npy'))
    y_test = np.load(opj(args.rawdir, 'y_test.npy'))

    #Do the preprocessing
    x_train= preprocess(x_train, args.mean, args.std)
    x_test = preprocess(x_test , args.mean, args.std)

    #Save data
    if not os.path.isdir(args.outdir):
    	os.makedirs(args.outdir)

    np.save(opj(args.outdir, 'x_train'), x_train)
    np.save(opj(args.outdir, 'y_train'), y_train)
    np.save(opj(args.outdir, 'x_test'), x_test)
    np.save(opj(args.outdir, 'y_test'), y_test)


    mlflow.set_tag('mean', args.mean)
    mlflow.set_tag('std', args.std)

    print("Saved Preprocessed Data into ", mlflow.active_run())



