import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys()) #x값
# dict_keys(['data', 'target', 'frame', 'target_names y값의 내용', 'DESCR', 'feature_names', 'filename경로'])
print(dataset.values())
print(dataset.target_names)
# 'setosa', 'versicolor', 'virginica'

x = dataset.data

y = dataset.target

df = pd.DataFrame(x, columns = dataset.feature_names) #x값

df.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
#y칼럼을 추가해보아요

df['Target'] = dataset.target

df.to_csv('../data/csv/iris_sklearn.csv', sep=',')