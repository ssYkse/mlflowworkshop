#!/bin/bash

### Download Pipe ###
dvc run \
	-f ./pipelines/mnist/download.dvc \
	-d src/download.py \
	-o data/mnist-raw \
	--no-exec \
	'mlflow run . -e get_data \
			-P type=number \
			-P outdir=./data/mnist-raw' 

### Explore Pipe ###
dvc run \
	-f ./pipelines/mnist/explore.dvc \
	-d src/explore.py \
	-d data/mnist-raw \
	-o params/auto/explore.yaml \
	--no-exec \
	'mlflow run . -e explore_data \
			-P in_data=./data/mnist-raw \
			-P paramsdir=params'

### Preprocess Pipe ###
dvc run \
	-f ./pipelines/mnist/preprocess.dvc \
	-d src/preprocess.py \
	-d data/mnist-raw \
	-d params/auto/explore.yaml \
	-o data/mnist-preprocessed \
	--no-exec \
	'mean=$(cat params/auto/explore.yaml | grep "mean" | sed "s/[^0-9\.]//g") && \
	std=$(cat params/auto/explore.yaml | grep "std" | sed "s/[^0-9\.]//g") && \
	mlflow run . -e preprocess \
			-P in_data=data/mnist-raw \
			-P mean=$mean \
			-P std=$std \
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
	-d params/auto/explore.yaml \
	-o models/packaged \
	--no-exec \
	'mean=$(cat params/auto/explore.yaml | grep "mean" | sed "s/[^0-9\.]//g") && \
	std=$(cat params/auto/explore.yaml | grep "std" | sed "s/[^0-9\.]//g") && \
	mlflow run . -e package \
			-P model_dir_in=models/raw \
			-P model_dir_out=models/packaged \
			-P mean=$mean \
			-P std=$std'

### Server Pipe ###
dvc run \
	-f ./pipelines/mnist/server.dvc \
	-d models/packaged \
	--no-exec \
	'mlflow models serve -p 5001 -m ./models/packaged'

### Docker Build Pipe ###
dvc run \
	-f ./pipelines/mnist/dockerbuild.dvc \
	-d models/packaged \
	--no-exec \
	'mlflow models build-docker -m ./models/packaged -n mnist_model'










