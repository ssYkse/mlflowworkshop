from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from os.path import join as opj
import mlflow
import mlflow.keras

import argparse

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('batch_size', type=int,
                    help='Batch size in training.')
parser.add_argument('epochs', type=int,
                    help='Nr. of epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Learning rate for training.')
parser.add_argument('input_dir', type=str,
                    help='artifact dir of a preprocessing run.')
args = parser.parse_args()

this_run = 0
if mlflow.active_run():
	this_run = mlflow.active_run()
else:
	this_run = mlflow.start_run()


for conv_1 in (8,16,24,32):
	with mlflow.start_run(nested=True) as child_run:
		p = mlflow.projects.run(
			uri="./../5/",
			entry_point="train",
			run_id = child_run.info.run_id,
			parameters={
				"batch_size": args.batch_size,
				"epochs": args.epochs,
				"nr_conv_1": conv_1,
				"lr": args.lr,
				"input_dir": args.input_dir
				},
			experiment_id=this_run.info.experiment_id,
			use_conda=False
		)
	
