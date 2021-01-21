import numpy as np
import pandas as pd

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2.모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam  #GD => Gradient Descent

optimizer = SGD(lr=1) #lr => learning rate
# <Adam>
#loss: 1.8800874101998488e-07 결과물: [[11.000813]] rl = 0.001
# loss: 6.963318945974253e-14 결과물: [[11.]] rl = 0.01
#oss: 3.8593527278862894e-05 결과물: [[11.003946]] rl = 0.1
#loss: nan 결과물: [[nan]] rl = 1
# <Adaelta>
# loss: 7.276543617248535 결과물: [[6.14774]] rl = 0.001
# loss: 1.3355738701648079e-05 결과물: [[10.993407]] rl = 0.01
# loss: 5.3182920964900404e-05 결과물: [[10.984306]] rl = 0.1
#loss: 0.33202001452445984 결과물: [[12.058738]]rl =1

# <Adamax>
# loss: 1.2354557554772327e-07 결과물: [[10.9997225]] rl =0.001 
#       3.480238233016797e-12 결과물: [[10.999998]] rl = 0.01
# loss: 5.920594503550092e-08 결과물: [[11.000133]] rl = 0.1
#loss: 7280141.0 결과물: [[3901.737]] rl = 1

#<Adagrad>
#loss: 4.336033271101769e-06 결과물: [[10.996992]] rl = 0.001
#loss: 8.854533916746732e-06 결과물: [[11.005882]] rl = 0.01
#loss: 234.81289672851562 결과물: [[-9.657964]] rl = 0.1
#loss: 1671470383104.0 결과물: [[-1603304.9]] rl = 1

#<RMSprop>
#loss: 0.0196085162460804 결과물: [[11.23581]] r1 = 0.001
#loss: 23.626113891601562 결과물: [[1.5550156]] rl = 0.01
#loss: 349829312.0 결과물: [[-27874.41]]rl = 0.1
#loss: 1.260244676264388e+21 결과물: [[-2.044275e+10]] rl = 1

#<SGD>
#loss: nan 결과물: [[nan]] rl=1
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4.평가, 예측
loss ,mse = model.evaluate(x,y, batch_size=1)
y_pred = model.predict([11])
print("loss:",loss, "결과물:",y_pred)