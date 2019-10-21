import argparse
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from os.path import join as opj
import numpy as np
import os

parser = argparse.ArgumentParser(description='Download either MNIST numbers or fashion')
parser.add_argument('outdir', type=str, 
                    help='Directory where to save the raw data.')
args = parser.parse_args()

#Make direcotry if doesn't exist
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

#Download the correct dataset
print("DOWNLOADING NUMBERS")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#save dataset
np.save(opj(args.outdir, 'x_train'), x_train)
np.save(opj(args.outdir, 'y_train'), y_train)
np.save(opj(args.outdir, 'x_test'), x_test)
np.save(opj(args.outdir, 'y_test'), y_test)
