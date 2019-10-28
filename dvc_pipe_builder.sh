#!/bin/bash

### Download Pipe ###
dvc run \
	-f ./pipelines/mnist/download.dvc \
	-d src/download.py \
	-o data/mnist-raw \
	--no-commit \
	--no-exec \
	'mlflow run . -e get_data \
			-P type=fashion \
			-P outdir=./data/mnist-raw' 

### Explore Pipe ###
dvc run \
	-f ./pipelines/mnist/expore.dvc \
	-d src/explore.py \
	-d data/mnist-raw \
	-d pipeline/mnist/download.dvc \
	--no-commit \
	--no-exec \
	'mlflow run . -e expore_data \
			-P in_data=./data/mnist-raw'

### Preprocess Pipe ###
dvc run \
	-f ./pipelines/mnist/preprocess.dvc \
	-d src/preprocess.py \
	-d data/mnist-raw \
	-o data/mnist-preprocessed \
	--no-commit \
	--no-exec \
	'mlflow run . -e preprocess  				\
			-P mean=33.318 				\ 
			-P std=78.567  				\
			-P in_data=./data/mnist-raw 		\
			-P outdir=./data/mnist-preprocessed'


### Train Pipe ###
dvc run \
	-f ./pipelines/mnist/train.dvc \
	-d src/train.py \
	-d data/mnist-preprocessed \
	-o models/raw \
	--no-commit \
	--no-exec \
	'mlflow run . -e train \
		      -P batch_size =32 \
		      -P epochs=3 \
		      -P lr=0.01 \
		      -P nr_conv_1=8 \
		      -P input_dir=data/mnist-preprocessed \
		      -P outdir=models/raw'














