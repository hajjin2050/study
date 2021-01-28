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
model = LinearSVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)# cv = cross_val_score/ 모델과 데이터를 엮어줌
#x,y를 적으면 트레인을 5등분 , x_train, y_train을 적으면 발리데이션데이터 5등분
print('scores:', scores) #scores: [1.         1.         0.83333333 0.96666667 1.        ]

#3.COMPILE
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])
# from tensorflow.keras.callbacks import EarlyStopping
# early_stoppig = EarlyStopping(monitor='loss', patience=30, mode='auto')
# model.fit(x, y, epochs=500,  validation_data=(x_val, y_val), batch_size=8 , callbacks=[early_stoppig], verbose=3)
model.fit(x, y)

#result = model.evaluate(x,y)
result = model.score(x,y) #바로 스코어 => 자동으로 에큐러시 추출
print("result :", result)

y_pred = model.predict(x[-5:-1])