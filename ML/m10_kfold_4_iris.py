import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = load_iris()
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

print(x.shape, y.shape)#(150,4) (105,3)


x_train, x_test,y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle= True, train_size =0.8)

kfold = KFold(n_splits = 5, shuffle=True)
    
#2.MODEL
models = [ LinearSVC(), SVC(),KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
for i in models:
    model = i
    model.fit(x_train, y_train)
    print(f'\n{i}')
#3.COMPILE
    model.fit(x, y)

#4.EVALUATE
#result = model.evaluate(x,y)
    y_pred = model.predict(x_test)
    result = model.score(x_test,y_test)
    print("model.score:",result)
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print('scores:', scores)
# LinearSVC()
# scores: [0.95833333 0.95833333 0.95833333 1.         1.        ]

# SVC()
# model.score: 0.9333333333333333
# scores: [0.91666667 0.91666667 1.         1.         0.95833333]

# KNeighborsClassifier()
# model.score: 0.9
# scores: [0.91666667 1.         0.95833333 1.         0.95833333]

# DecisionTreeClassifier()
# model.score: 1.0
# scores: [1.         0.91666667 0.95833333 1.         0.95833333]

# RandomForestClassifier()
# model.score: 1.0
# scores: [0.91666667 1.         0.91666667 1.         1.        ]