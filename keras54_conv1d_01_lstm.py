#Conv1d로 완성하시오
import numpy as np
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

# print(x.shape)  #(13, 3)
# print(y.shape)  #(13,)

print(np.max(x), np.min(x))

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train =train_test_split(x, y, train_size= 0.6, shuffle=True, random_state = 112)
x_val, x_train, y_val, y_train =train_test_split(x_train, y_train, train_size= 0.6, shuffle=True, random_state = 112)


x_train = x_train.astype('float32')/60.
x_val = x_val.astype('float32')/60.
x_test = x_test.astype('float32')/60.

print(x_train[0].shape)
print(x_train[1].shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPooling1D,Dropout,Flatten

model = Sequential()
model.add(Conv1D(filters=400, kernel_size=1, padding = 'same', strides=2
                        ,input_shape=(3,1)))
model.add(MaxPooling1D(pool_size = 1))
model.add(Dropout(0.2))
model.add(Conv1D(300, (1), padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(100, (1), padding='same'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = 'val_loss', patience=5, mode='min')
model.fit(x_test, y_test, epochs=600, batch_size=8, verbose=3, validation_data=(x_val, y_val), callbacks=[es])

#4.평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=69)
print('loss, acc: ', loss, acc)

y_pred = model.predict(x_test)
