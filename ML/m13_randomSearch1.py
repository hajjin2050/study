import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)

# dataset = pd.read_csv("C;data/csv/iris_sklearn.csv", hearder= 0, index_col = 0)
# x = dataset.iloc[:,:-1]
# y = datset.iloc[:,-1]

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

kfold = KFold(n_splits = 5, shuffle=True)
    
#dictionary
parameters = [
    {'n_estimators': [100,200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs':[-1]}
]

#2.MODEL
model = RandomizedSearchCV(SVC(), parameters,cv=kfold) #나는 svc모델을 RandomizedSearchCV방식으로 쓰겠다

scores = cross_val_score(model, x_train, y_train, cv=kfold)# cv = cross_val_score/ 모델과 데이터를 엮어줌

#3.COMPILE
model.fit(x_train, y_train)
#4.EVALUATE
print("최적의 매개변수:", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종정답률:",accuracy_score(y_test, y_pred))
print("scores:", scores)
