import mlflow.pyfunc
import numpy as np
import argparse
import mlflow.keras
import keras
from os.path import join as opj
import base64
#from preprocess import preprocess

parser = argparse.ArgumentParser(description='Train a Keras model for MNIST.')
parser.add_argument('model_dir_in', type=str,
                    help='model dir of a training run.')
parser.add_argument('model_dir_out', type=str,
                    help='model dir where packaged model is to be saved.')
parser.add_argument('mean', type=float,
                    help='mean vor normalization in preprocessing.')
parser.add_argument('std', type=float,
                    help='std vor normalization in preprocessing.')
args = parser.parse_args()

def preprocess(x_train, mean, std):
    #Normalize data
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_train = x_train - mean
    x_train = x_train / std

    return x_train

class MyPipeModel(mlflow.pyfunc.PythonModel):
    import keras
    """
    A pyfunc model which takes as path a path to the Keras model directory in the artifacts dir.
    This Class includes preprocessing steps.
    """
    def __init__(self, path):
        self.keras_path= path
        self.model = mlflow.keras.load_model(path)
        self.pre = preprocess
        self.mean = args.mean
        self.std = args.std

    
    def predict(self, context, model_input):
        print("Prediction!")

        x = model_input.to_numpy()
        x = self.pre(x.reshape(1,28,28), self.mean, self.std)

        results = self.model.predict_classes(x)

       # return results
        return results

    def save(self, model_path):
        conda = opj(self.keras_path, 'conda.yaml')
        mlflow.pyfunc.save_model(path=model_path, python_model=self, conda_env=conda)

if __name__ == '__main__':
    mypipe = MyPipeModel(args.model_dir_in)
    mypipe.save(args.model_dir_out)
