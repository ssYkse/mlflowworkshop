import mlflow 
from sklearn.linear_model import LogisticRegression
import numpy as np

import mlflow.sklearn

X = np.array([-2, -1, 0, 1, 2, 3 ,4]).reshape(-1,1)
y = np.array([0,0,0,1,1,1,2])

lr = LogisticRegression()
lr.fit(X, y)
score = lr.score(X,y)

mlflow.log_metric("score", score)
mlflow.sklearn.log_model(lr, "my_lr_model")


