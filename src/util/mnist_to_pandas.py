import pandas
import numpy as np
from os.path import join as opj
import os
import argparse

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('input_dir', type=str,
                    help='artifact dir of a preprocessing run.')
parser.add_argument('i', type=int, 
                    help='which image to return')
args = parser.parse_args()

x_test = np.load(opj(args.input_dir, 'x_test.npy'))

x = x_test[args.i,:,:,0]

df = pandas.DataFrame(x)

df.to_json('./example.json', orient='split')
