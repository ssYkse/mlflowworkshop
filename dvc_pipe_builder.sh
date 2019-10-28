#!/bin/bash

### Download Pipe ###
dvc run \
	-f ./pipelines/mnist/download.dvc \
	-d src/download.py \
	-o data/mnist-raw \
	--no-exec \
	'mlflow run . -e get_data \
			-P type=fashion \
			-P outdir=./data/mnist-raw' 

### Explore Pipe ###
dvc run \
	-f ./pipelines/mnist/explore.dvc \
	-d src/explore.py \
	-d data/mnist-raw \
	--no-exec \
	'mlflow run . -e explore_data \
			-P in_data=./data/mnist-raw'

### Preprocess Pipe ###
dvc run \
	-f ./pipelines/mnist/preprocess.dvc \
	-d src/preprocess.py \
	-d data/mnist-raw \
	-o data/mnist-preprocessed \
	--no-exec \
	'mlflow run . -e preprocess \
			-P in_data=data/mnist-raw \
			-P mean=33.318 \
			-P std=78.567 \
			-P outdir=data/mnist-preprocessed'


### Train Pipe ###
dvc run \
	-f ./pipelines/mnist/train.dvc \
	-d src/train.py \
	-d data/mnist-preprocessed \
	-o models/raw \
	--no-exec \
	'mlflow run . -e train \
		      -P batch_size=32 \
		      -P epochs=3 \
		      -P lr=0.01 \
		      -P nr_conv_1=8 \
		      -P input_dir=data/mnist-preprocessed \
		      -P outdir=models/raw'


### Package Pipe ###
dvc run \
	-f ./pipelines/mnist/package.dvc \
	-d src/package_model.py \
	-d models/raw \
	-o models/packaged \
	--no-exec \
	'mlflow run . -e package \
			-P model_dir_in=models/raw \
			-P model_dir_out=models/packaged \
			-P mean=33.318 \
			-P std=78.567'

### Server Pipe ###
dvc run \
	-f ./pipelines/mnist/server.dvc \
	-d models/packaged \
	--no-exec \
	--always-changed \
	'mlflow models serve -p 5001 -m ./models/packaged'

### Docker Build Pipe ###
dvc run \
	-f ./pipelines/mnist/dockerbuild.dvc \
	-d models/packaged \
	--no-exec \
	--always-changed \
	'mlflow models build-docker -m ./models/packaged -n mnist_model'










