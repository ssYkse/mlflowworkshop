import argparse
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from os.path import join as opj
import numpy as np
import os

parser = argparse.ArgumentParser(description='Download either MNIST numbers or fashion')
parser.add_argument('type', type=str, 
                    help='[fashion/number]')
parser.add_argument('outdir', type=str, 
                    help='Directory where to save the raw data.')
args = parser.parse_args()

#Make direcotry if doesn't exist
if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

#Download the correct dataset
if args.type == 'number':
    print("DOWNLOADING NUMBERS")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif args.type == 'fashion':
    print("DOWNLOADING FASHION")
    #((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    import mnist_reader
    x_train, y_train = mnist_reader.load_mnist('data/', kind='train')
    x_test, y_test = mnist_reader.load_mnist('data/', kind='t10k')
    x_train = x_train.reshape(x_train.shape[0], 28,28)
    x_test = x_test.reshape(x_test.shape[0], 28,28)
else:
    print("Type not known. Either [fashion] or [number].")

#save dataset
np.save(opj(args.outdir, 'x_train'), x_train)
np.save(opj(args.outdir, 'y_train'), y_train)
np.save(opj(args.outdir, 'x_test'), x_test)
np.save(opj(args.outdir, 'y_test'), y_test)
