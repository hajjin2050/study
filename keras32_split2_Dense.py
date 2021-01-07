import numpy as np
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)


dataset = split_x(a, size)
print("==================")
print(dataset)


x = dataset[:,:4]  # dataset[행, 렬]  dataset[0:6, 0:4]
y = dataset[:,4]  # dataset[0:6, 4]
print(y.shape)
print(x.shape)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(4,))) #(행, 열, 몇개씩자르는지)
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(1))

model.summary() 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x, y, epochs=2000, batch_size=16, callbacks=[early_stopping])


# 4. 평가, 예측
loss = model.evaluate(x, y, batch_size=16)
print("loss:", loss)
x_pred = np.array(range(7,11))
x_pred = x_pred.reshape(1, 4)
y_pred = model.predict(x_pred)
print(y_pred)


#  ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 4 but received input with shape [None, 1]
#