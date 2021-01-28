import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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


# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical #이게텐서플로우 1.0방식. 이것도 가능하긴 하다.

# y = to_categorical(y)
# y_train= to_categorical(y_train)
# y_val= to_categorical(y_val)
# y_test= to_categorical(y_test)

print(x.shape)#(150,4)
print(y.shape) # (105,3)
    


#2.MODEL
model = RandomForestClassifier()





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
print("y_pred:", y_pred)
print(y[-5:-1])

# LinearSVC
# result : 0.9666666666666667
# SVC
# result : 0.9733333333333334
# KNeighborsClassifier
# result : 0.9666666666666667
# DecisionTreeClassifier
# result : 1.0
# RandomForestClassifier
# result : 1.0

#tensorflow
#acc: ???

#성능 머선129...성능 너무 좋다