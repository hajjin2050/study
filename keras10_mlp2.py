import numpy as np


#1.데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))
print(x.shape)           #(3,100) 
print(y.shape)           #(100,)

x = np.transpose(x)
print(x) 
print(x.shape)     #(100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle = True, random_state=66 )


print(x_train.shape)      #(80, 3)
print(y_train.shape)      #(80,)



#print(x.reshape(10,2))     #(10,) -> (2, 10)
    #(10,) -> (2, 10)


#2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(1))


#3.compile
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4.평가예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test) # y_test 랑 같이 묶여있는값

#scikit learn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt : 루트를 씌우기.
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)