# keras23_3을 카피해서
#LSTM층을 두개로 만들것

#model.add(LSTM(10,input_shape=(3,1)))
#model.add(LSTM(10))

import numpy as np
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

# print(x.shape)
# print(y.shape)
print(x.shape[0])
print(x.shape[1])

# x = x.reshape(13, 3, 1)아래와 같음
x = x.reshape(x.shape[0],x.shape[1],1)
#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(80,activation = 'linear', input_shape=(3,1), return_sequences=True))
#return_sequences의 default갓은 False/  Dense는 2차원이니까 그전까지는 3차원 유지 
model.add(LSTM(70))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 80)             26240   #아웃풋 노드의 수가 input_dim이 됨
_________________________________________________________________
lstm_1 (LSTM)                (None, 70)                42280
_________________________________________________________________
dense (Dense)                (None, 55)                3905
_________________________________________________________________
dense_1 (Dense)              (None, 40)                2240
_________________________________________________________________
dense_2 (Dense)              (None, 35)                1435
_________________________________________________________________
dense_3 (Dense)              (None, 30)                1080
_________________________________________________________________
dense_4 (Dense)              (None, 25)                775
_________________________________________________________________
dense_5 (Dense)              (None, 10)                260
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 11
=================================================================
Total params: 78,226
Trainable params: 78,226
Non-trainable params: 0
'''
#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600, batch_size=8, verbose=3)

#4.평가 예측
loss = model.evaluate(x,y)
print("loss:",loss)

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3, 1) 

result = model.predict(x_pred)
print("result:", result)
