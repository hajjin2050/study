import numpy as np
#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape) 

x = np.transpose(x)
print(x) 
print(x.shape)     #(10,2)

#print(x.reshape(10,2))     #(10,) -> (2, 10)
    #(10,) -> (2, 10)


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(1))


#3.compile
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)

#4.평가예측
loss, mae = model.evaluate(x, y)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x)
'''
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
'''