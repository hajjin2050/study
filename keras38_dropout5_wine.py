from sklearn.datasets import load_wine
import numpy as np

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split (
    x, y, train_size= 0.6, random_state = 66
)
from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train = train_test_split (
    x_train, y_train, train_size= 0.6, random_state = 66
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
scaler.transform(x_val)

#2.MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(10,))
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
output = Dense(1)(dense1)

model = Model(inputs = input1 ,outputs= output)
model.summary()

#3.compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #loss='binary_crossentropy'->이진분류일때 사용!!
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=earlystopping)



#4.EVALUATE
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss:",loss)

y_pred = model.predict(x)