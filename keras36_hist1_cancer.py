#hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

import numpy as np
from sklearn.datasets import load_breast_cancer
#1.DATA
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)


#2.MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(200,activation = 'relu',input_shape=(30,))) #'relu' - > 0~무한으로 수렴
model.add(Dense(147))
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(76))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))

#3.compile
model.compile(loss='mse', optimizer='adam', metrics=['acc']) #loss='binary_crossentropy'->이진분류일때 사용!!
hist = model.fit(x, y, epochs=100, batch_size=10,verbose=3, validation_split = 0.2,  )
print(hist)
print(hist.history.keys())

#4.EVALUATE
loss = model.evaluate(x, y)
print("loss:",loss)

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



