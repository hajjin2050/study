#1:多 

import numpy as np
#1.데이터
x1 = np.array([range(100), range(301, 401),range(1, 101)])
y1 = np.array([range(711, 811), range(1, 101),range(201, 301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)

y1 = np.transpose(y1)
y2 = np.transpose(y2)



from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1,y2, shuffle=False, train_size = 0.8
)
'''
from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=False, train_size = 0.8
)
'''
#2.모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

#모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
output1 = Dense(2)(dense1)

#모델2
#input1 = Input(shape=(3,))
#dense2 = Dense(10, activation='relu')(input1)
#dense2 = Dense(5, activation='relu')(dense2)
#dense2 = Dense(5, activation='relu')(dense2)
#dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(2)(dense2)
'''
#모델변형 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
'''
#모델 분기1
output1 = Dense(30)(dense1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

#모델 분기2
output2 = Dense(30)(dense1)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

#모델 선언
model = Model(inputs= input1, 
              outputs=[output1, output2])
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train,[y1_train, y2_train], epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가 , 예측
loss = model.evaluate(x1_test,[y1_test, y2_test], batch_size=1)

#loss: [3198.59716796875<전체로스(1번쨰loss+2번째loss 혹은 1번째 metrics+2번째 metrics)>, 2104.79638671875<1번째 loss값>, 1093.80078125<2번째 loss값>, 2104.79638671875<1번째 metrcis값>, 1093.80078125<2번째 metrics값>]
print("model.metrics_names : ", model.metrics_names)
print("loss:", loss)


y1_predict, y2_predict = model.predict([x1_test])

print("=====================" )
print("y1_predict : \n", y1_predict )
print("=====================" )
print("y2_predict : \n", y2_predict )

#scikit learn
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict))  #sqrt : 루트를 씌우기.

RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)
RMSE = (RMSE1+RMSE2)/2

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE :", RMSE)


from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2 = (r2_1 + r2_2)/2


print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 :", r2)
