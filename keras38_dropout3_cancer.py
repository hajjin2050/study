import numpy as np
from sklearn.datasets import load_breast_cancer
#1.DATA
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print('x.shape:',x.shape)  

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)
from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train = train_test_split(
    x_train, y_train, train_size = 0.8, shuffle=True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

print(x.shape)
print(y.shape)

#2.MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(30,))
dense1 = Dense(200,activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(150,activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(100,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(80,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(50,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(30,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
output = Dense(2)(dense1)

model = Model(inputs = input1 ,outputs= output)
model.summary()

#3.compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #loss='binary_crossentropy'->이진분류일때 사용!!
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=1000,  batch_size=10, validation_data=(x_val, y_val), verbose= 3, callbacks=[earlystopping])



#4.EVALUATE
loss = model.evaluate(x_test, y_test,batch_size=2)
# print("loss:",loss)


y_pred = model.predict(x[:5])
print('y_pred:',y_pred)
print('y[:5], y[:5]')

#loss: [0.8823392391204834, 0.8549450635910034]

