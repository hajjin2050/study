import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape)

# print(np.max(x), np.min(y))
# print(dataset.feature_names)
# print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6 , randeom_state=121
)

#MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shpae=(10,),activation='relu')
dense1 = Dense(100,activation='relu')(input1)
dense1 = Dense(80,activation='relu')(dense1)
dense1 = Dense(50,activation='relu')(dense1)
dense1 = Dense(30,activation='relu')(dense1)
output = Dense(1,activation='relu')(dense1)

model = Model(inputs = input1, outputs = output)

##COMPILE
model.compile(loss='mae', optimizer='adam',)

