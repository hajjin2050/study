import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = load_wine()
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
# model.score: 0.7777777777777778
# scores: [0.93103448 0.65517241 0.85714286 0.78571429 0.85714286]

# SVC()
# model.score: 0.7222222222222222
# scores: [0.65517241 0.65517241 0.71428571 0.60714286 0.60714286]

# KNeighborsClassifier()
# model.score: 0.7777777777777778
# scores: [0.5862069  0.5862069  0.71428571 0.60714286 0.71428571]

# DecisionTreeClassifier()
# model.score: 1.0
# scores: [0.89655172 0.86206897 0.92857143 0.85714286 0.96428571]

# RandomForestClassifier()
# model.score: 1.0
# scores: [0.96551724 1.         0.92857143 1.         0.96428571]