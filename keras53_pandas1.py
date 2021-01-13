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
# x = dataset['data']  + # df = pd.DataFrame(x, columns = dataset['feature_names'])
y = dataset.target
'''
print(x)
print(y)
print(x.shape, y.shape) #(150, 4) (150,)
print(type(x),type(y)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

#numpy--> pandas
df = pd.DataFrame(x, columns = dataset.feature_names) #x값
# df = pd.DataFrame(x, columns = dataset['feature_names']) #x값
print(df)

#   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   => HEAD (not data)
# 0                  5.1               3.5                1.4               0.2  |
# 1                  4.9               3.0                1.4               0.2  |
# 2                  4.7               3.2                1.3               0.2  | 
# 3                  4.6               3.1                1.5               0.2  |
# 4                  5.0               3.6                1.4               0.2  |
# ..                 ...               ...                ...               ...  |  ==> INDEX(not data)
# 145                6.7               3.0                5.2               2.3  |
# 146                6.3               2.5                5.0               1.9  |
# 147                6.5               3.0                5.2               2.0  |
# 148                6.2               3.4                5.4               2.3  |
# 149                5.9               3.0                5.1               1.8  |
print(df.shape) #[150 rows x 4 columns](150, 4)
print(df.columns)
print(df.index)  #명시해주지않으면 자동indexing
# #Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#        'petal width (cm)'],
#       dtype='object')
# RangeIndex(start=0, stop=150, step=1)

print(df.head())
#df[:5]
print(df.tail())#df[:-5]
print(df.info())
print(df.describe())

df.columns = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
print(df.columns)
print(df.info())
print(df.describe())

#y칼럼을 추가해보아요
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())

print(df.shape) #(150,5)
print(df.columns)
print(df.index)
print(df.tail())

print(df.info())
print(df.isnull())
print(df.isnull().sum())
print(df.describe())

#상관계수
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set (font_scale=1.2)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

#도수 분포도
plt.figure(figsize=(10, 6))
plt.subplot(2,2,1)
plt.hist(x = 'sepal_length', data = df)
plt.title('sepal_length')

plt.subplot(2, 2, 2)
plt.hist(x = 'sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2, 2, 3)
plt.hist(x = 'petal_length', data = df)
plt.title('petal_length')

plt.subplot(2, 2, 4)
plt.hist(x = 'petal_width', data = df)
plt.title('petal_width')

plt.show()
'''