import numpy as np
from sklearn.datasets import load_breast_cancer
#1.DATA
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print('x.shape:',x.shape)   #(569,30)
print('y.shape:',y.shape)   #(569,)
# print(x[:5])
# print(y)

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)


#2.MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30,activation = 'relu',input_shape=(30,))) #'relu' - > 0~무한으로 수렴
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

#3.compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #loss='binary_crossentropy'->이진분류일때 사용!!
model.fit(x, y, epochs=200, validation_split=0.2, batch_size=8)



#4.EVALUATE
loss = model.evaluate(x, y)
print("loss:",loss)
y[-5:-1]
y_pred = model.predict(x[-5:-1])
print('y_pred:',y_pred)
print("y[-5:-1]:",y[-5:-1])