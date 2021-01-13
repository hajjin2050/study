#인공지능계의 hello world라 불리는 mnist!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(600000, 28, 28) (600000,)
print(x_test.shape, y_test.shape)    #(100000, 28, 28) (100000,)

from sklearn.model_selection import train_test_split
x_val, y_val_ , x_train,y_train = train_test_split(
    x_train, y_train, train_size = 0.6, shuffle=True, random_state= 122
) 

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]* x_test.shape[2], 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#MODELING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=400, kernel_size=2, padding = 'same', strides=2
                        ,input_shape=(784,1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv1D(300, 2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(100, 2, padding='same'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=30 , mode='auto')

model.compile(loss = 'mse', optimizer='adam',metrics=['mae'])
model.fit(x_test, y_test , epochs=100,validation_data=(x_val, y_val_) ,batch_size=64, verbose=3 , callbacks=[es])


#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)