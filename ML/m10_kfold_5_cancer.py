import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

print(x.shape, y.shape)#(150,4) (105,3)


x_train, x_test,y_train, y_test = train_test_split(
    x, y, random_state=77, shuffle= True, train_size =0.8)

kfold = KFold(n_splits = 5, shuffle=True)
    
#2.MODEL
models = [ LinearSVC(), SVC(),KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()
            ,LogisticRegression()]
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
# model.score: 0.8333333333333334   
# scores: [0.89010989 0.92307692 0.82417582 0.93406593 0.92307692]

# SVC()
# model.score: 0.9210526315789473
# scores: [0.86813187 0.91208791 0.9010989  0.93406593 0.89010989]

# KNeighborsClassifier()
# model.score: 0.956140350877193
# scores: [0.86813187 0.96703297 0.93406593 0.93406593 0.91208791]

# DecisionTreeClassifier()
# model.score: 1.0
# scores: [0.94505495 0.9010989  0.9010989  0.95604396 0.96703297]

# RandomForestClassifier()
# model.score: 1.0
# scores: [0.96703297 0.96703297 0.95604396 0.97802198 0.98901099]

# LogisticReression()
# model.score: 0.9385964912280702
# scores: [0.97802198 0.92307692 0.94505495 0.92307692 0.93406593]