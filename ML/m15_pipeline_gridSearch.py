#gridsearch와 pipeline을 이어줌
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#dictionary 파라미터 튜닝하는거임 
# parameters = [
#     {"svc__C": [1, 10, 100, 1000], "svc__kernel":["linear"]},
#     {"svc__C": [1, 10, 100], "svc__kernel":["rbf"], "svc__gamma":[0.001, 0.0001]},
#     {"svc__C": [1, 10, 100, 10000], "svc__kernel":["sigmoid"], "svc__gamma":[0.001, 0.0001]} #SVC 이름 따서 그대로 들어감
# ]
parameters = [
    {"mal__C": [1, 10, 100, 1000], "mal__kernel":["linear"]},
    {"mal__C": [1, 10, 100], "mal__kernel":["rbf"], "mal__gamma":[0.001, 0.0001]},
    {"mal__C": [1, 10, 100, 10000], "mal__kernel":["sigmoid"], "mal__gamma":[0.001, 0.0001]} #SVC 이름 따서 그대로 들어감
]
#2.모델
pipe = Pipeline([("scaler", MinMaxScaler()),('mal', SVC())])
# pipe = make_pipeline(MinMaxScaler(),SVC())

# model = GridSearchCV(pipe, parameters,cv=5)
model = RandomizedSearchCV(pipe, parameters,cv=5)

model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results) #1.0