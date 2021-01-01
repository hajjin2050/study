#다 : 1 mlp (keras10_mlp2.py copy)
import numpy as np

#1.데이터
x = np.array([range(500), range(600), range(400)])
y = np.array(range(711, 811))
print(x.shape)
print(y.shape)


x = np.transpose(x)
print(x)
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle = True, randome_state=1) 
print(x_train.shape)
print(y_train.shape)

#2.모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(5)(dense1)
outputs = Dense(1)(dense2)
model = Model(inputs = input1, outputs = outputs)
model.summary()

