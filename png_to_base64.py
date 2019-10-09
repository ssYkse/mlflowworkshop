import cv2
import numpy as np
import pandas as pd
import argparse
import base64

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('input_png', type=str,
                    help='png to be converted to numpy > pandas > json')
args = parser.parse_args()

img = cv2.imread(args.input_png, cv2.IMREAD_GRAYSCALE)

encoded = base64.b64encode(
    np.arange(28*28, dtype=np.float64))

f = open('bytes.txt', 'w')
f.write(str(encoded))