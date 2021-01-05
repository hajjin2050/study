#코딩하시오!! LSTM
#나는 80을 원하고있다

import numpy as np
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape)
print(y.shape)

x = x.reshape(13, 3, 1)
#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU

model = Sequential()
model.add(GRU(80,activation = 'linear', input_shape=(3,1)))
model.add(Dense(70))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


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

#LSTM
#loss: 0.27925413846969604
#result: [[81.72403]]

#SimpleRNN
# loss: 5.287312601631733e-11
# result: [[80.000015]]

# GRU
# loss: 0.006781375966966152
# result: [[80.097275]]