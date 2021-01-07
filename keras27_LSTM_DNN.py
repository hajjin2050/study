#keras23_LSTM3_scale을  함수형으로 코딩
#코딩하시오!! LSTM
import numpy as np
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,train_size = 0.8,random_state = 121)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.2,random_state = 121)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
scaler.transform(x_val)

# x = x.reshape(13, 3)

#2.모델링

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape =(3,))
dense1 = Dense(100, activation='relu')(input1)
dense1 = Dense(80)(dense1)
dense1 = Dense(65)(dense1)
dense1 = Dense(35)(dense1)
dense1 = Dense(10)(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)

#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600, batch_size=8, verbose=3)

#4.평가 예측
loss = model.evaluate(x,y)
print("loss:",loss)

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3)

result = model.predict(x_pred)
print("result:", result)

