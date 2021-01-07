#LSTM 모델을 구성하시오

import numpy as np
a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1 ):                 #행
        subset = seq[i : (i+size)]                        #열
        aaa.append([item for item in subset])   #
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

print("=========================")
print(dataset)
x = dataset[:,:4]  #[행, 열] = [0:6(모든 행), 0:4] = [:, :4] 
y = dataset[:,4]  #[행, 열] = [0:6(모든 행), 4] = [:, 4] = [:, -1] = [0:-1, -1]
x_pred = dataset[-1,1:]
print("x.shape :", x.shape)
print("y.shape :", y.shape)
print("x_pred.shape :", x_pred.shape)


x = x.reshape(6,4,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(80, activation='relu', input_shape=(4,1)))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y, epochs=100, batch_size=5, verbose=3)

loss = model.evaluate(x,y, batch_size=16)
print("loss:",loss)


x_pred = dataset[-1,1:]
# print(b.shape) #(4,)
x_pred = x_pred.reshape(1,4,1)
# print(x_pred.shape) #(1,4,1)
y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# loss: 0.010317912325263023
# y_pred:  [[10.9489565]]