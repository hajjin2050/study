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

# x = x.reshape(178, 13, 1)

#2.MODEL
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(Dense(100,activation='relu', input_shape=(13,)))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#COMPILE
model.compile(loss='mse', optimizer='adam', metrics= 'acc')
from tensorflow.keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='loss', patience=30,mode = 'auto')
hist = model.fit (x, y, epochs=100, validation_data=(x_val, y_val),verbose=3,callbacks=[earlystop])

#EVALUATE
loss = model.evaluate(x_test, y_test, batch_size=10)
print("loss:", loss)

print(hist.history['loss'])

#그래프
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) #legend = >주석, 어떤그래프인지 
plt.show()
