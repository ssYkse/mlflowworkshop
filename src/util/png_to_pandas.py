import cv2
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('input_png', type=str,
                    help='png to be converted to numpy > pandas > json')
args = parser.parse_args()

img = cv2.imread(args.input_png, cv2.IMREAD_GRAYSCALE)

print(img.shape)

df = pd.DataFrame(img)

df.to_json('./example.json', orient='split')
