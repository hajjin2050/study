# model = RandomForestclassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)

print(x.shape, y.shape)#(150,4) (105,3)

kfold = KFold(n_splits = 5, shuffle=True)
import datetime

date_now1 = datetime.datetime.now()
date_time = date_now1.strftime("%m월%d일_%H시%M분%S초")
print("start time: ",date_time)
    
#dictionary
parameters = [
    {'n_estimators': [100,200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5,10]},
    {'n_jobs':[-1]}
]


#2.MODEL
model = GridSearchCV(RandomForestClassifier(), parameters,cv=kfold) #나는 svc모델을 gridsearchcv방식으로 쓰겠다

scores = cross_val_score(model, x_train, y_train, cv=kfold)# cv = cross_val_score/ 모델과 데이터를 엮어줌

#3.COMPILE
model.fit(x_train, y_train)
#4.EVALUATE
print("최적의 매개변수:", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종정답률:",accuracy_score(y_test, y_pred))
print("scores:", scores)

date_now2 = datetime.datetime.now()
date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)
print("걸린시간 : ",(date_now2-date_now1))

# start time:  01월28일_22시35분36초
# 최적의 매개변수: RandomForestClassifier()
# 최종정답률: 0.9333333333333333
# scores: [0.95833333 0.875      0.95833333 1.         0.95833333]
# End time:  01월28일_22시36분23초
# 걸린시간 :  0:00:47.160518