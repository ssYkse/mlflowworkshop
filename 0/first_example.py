import mlflow
import tempfile
import os

mlflow.set_tag('my_tag', 'I am the tag')

learning_rate = 10
mlflow.log_param("LR", learning_rate)
for i in range(10):
	print("Amazing Training ongoing!")
	mlflow.log_metric('loss', i)
	mlflow.log_metric('amazingnes', 1/(i+1))


dirpath = tempfile.mkdtemp()
with open(os.path.join(dirpath , 'testmodel.txt'), 'w') as f:
	f.write("hello")

print("Logging Files")
mlflow.log_artifacts(dirpath, artifact_path="experiment_data")
