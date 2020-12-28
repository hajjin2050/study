
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array
# np. array()
# array

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])
x_pred = array([16,17,18])

#2. 모델구성
model = Sequential()
model.add(Dense(3131, input_dim=1, activation= 'relu'))
model.add(Dense(11))
model.add(Dense(81))
model.add(Dense(874))
model.add(Dense(1254))
model.add(Dense(115))
model.add(Dense(2101))
model.add(Dense(266))
model.add(Dense(2324))
model.add(Dense(202))
model.add(Dense(2023))
model.add(Dense(520))
model.add(Dense(290))
model.add(Dense(230))
model.add(Dense(250))
model.add(Dense(2740))
model.add(Dense(17))
model.add(Dense(420))
model.add(Dense(320))
model.add(Dense(210))
model.add(Dense(250))
model.add(Dense(250))
model.add(Dense(205))
model.add(Dense(250))
model.add(Dense(2240))
model.add(Dense(205))
model.add(Dense(206))
model.add(Dense(205))
model.add(Dense(202))
model.add(Dense(205))
model.add(Dense(520))
model.add(Dense(81))
model.add(Dense(874))
model.add(Dense(1254))
model.add(Dense(115))
model.add(Dense(2101))
model.add(Dense(266))
model.add(Dense(2354))
model.add(Dense(2102))
model.add(Dense(2023))
model.add(Dense(520))
model.add(Dense(2905))
model.add(Dense(230))
model.add(Dense(250))
model.add(Dense(2740))
model.add(Dense(1755))
model.add(Dense(420))
model.add(Dense(320))
model.add(Dense(210))
model.add(Dense(250))
model.add(Dense(250))
model.add(Dense(205))
model.add(Dense(250))
model.add(Dense(20))
model.add(Dense(205))
model.add(Dense(204))
model.add(Dense(2055))
model.add(Dense(202))
model.add(Dense(205))
model.add(Dense(520))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=303, batch_size=1, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size = 1)
print("mse,mae: ", results)

y_predict = model.predict(x_test)
#print("y_predict :", y_predict)

#scikit learn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt : 루트를 씌우기.
print("RMSE : ",RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)