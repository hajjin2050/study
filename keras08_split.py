from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
from numpy import array
# np. array()
# array

# 1. 데이터
x = np.array(range(1, 101))
#x = np.array(range(100))
y = np.array(range(101, 201))

x_train = x[:60]  # 순서 0번째부터 59번째까지 :::: 1-60
x_val = x[60:80]  # 61~80 :
x_test = x[80:]  # 81 ~ 100
# 리스트의 슬라이싱

y_train = y[:60]  # 순서 0번째부터 59번째까지 :::: 1-60
y_val = y[60:80]  # 61~80 :
y_test = y[80:]  # 81 ~ 100
# 리스트의 슬라이싱

# 2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(30))
model.add(Dense(42))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss: ", loss)

y_predict = model.predict(x_test)
#print("y_predict :", y_predict)

# scikit learn


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  # sqrt : 루트를 씌우기.


print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
