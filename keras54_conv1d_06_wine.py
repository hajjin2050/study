import numpy as np

#1. 데이터
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)
x_val = x_scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
print(x_train.shape[1])


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=300, kernel_size=1, padding='same', strides=1, input_shape=(10,1)))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(300,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(180,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(130,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=30, mode='atuo')
model.fit(x_train, y_train, epochs=800, batch_size=69, validation_data=(x_val, y_val), verbose=3, callbacks=[stop])


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=69)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test)

# r2, rmse
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_pred,y_test)
rmse = mean_squared_error(y_pred,y_test)**0.5
print('rmse : ',rmse)
print('r2 : ',r2)