import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation= 'linear'))
model.add(Dense(3, activation= 'linear'))
model.add(Dense(4, name='helloworld'))
model.add(Dense(10))
model.add(Dense(18))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(800))
model.add(Dense(1))

model.summary()

#실습 2 + 과제
#ensemble 1, 2, 3, 4 에 대해 서머리를 계산하고
#이해한 것을과제로 제출할 것
#layer를 만들떄 'name'에 대해 확인하고 설명할 것    
# name을 만드시 써야할 떄가 있다. 그때를 말할 것