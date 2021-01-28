import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression #회기모델같지만 분류모델임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size= 0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#2.MODEL
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()

#3.COMPILE
metrics=['accuracy_score']
model.fit(x, y)

#4.EVALUATE
#result = model.evaluate(x,y)
y_pred = model.predict(x)
result = model.score(x,y)
print("model.score:",result)
acc = accuracy_score(y,y_pred)
print("accuracy_score:",acc)
#LinearSVC
# model.score: 0.9244288224956063
# accuracy_score: 0.9244288224956063

# SVC
# model.score: 0.9226713532513181
# accuracy_score: 0.9226713532513181

# KNeighborsClassifier
# model.score: 0.9472759226713533
# accuracy_score: 0.9472759226713533

# DecisionTreeClassifier
# model.score: 1.0
# accuracy_score: 1.0

# RandomForestClassifier
# model.score: 1.0
# accuracy_score: 1.0

#tensorflow
#acc: ???