import numpy as np
#1.데이터
x1 = np.array([range(100), range(301, 401),range(1, 101)])
y1 = np.array([range(711, 811), range(1, 101),range(201, 301)])

x2 = np.array([range(101,201), range(411,511), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])
y3 = np.array([range(601, 701), range(811, 911), range(1100, 1200)])


x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)





from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, y1,x2, y2, y3, shuffle=False, train_size = 0.8
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
dense1 = Dense(8, activation='relu')(input1)
dense1 = Dense(6, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
#output1 = Dense(3)(dense1)

#모델2
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense1 = Dense(8, activation='relu')(input1)
dense2 = Dense(6, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(3)(dense2)


#모델변형 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

#모델 분기1
output1 = Dense(30)(middle1)
output1 = Dense(20)(output1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

#모델 분기2
output2 = Dense(30)(middle1)
output2 = Dense(20)(output2)
output2 = Dense(10)(output2)
output2 = Dense(3)(output2)

#모델 분기2
output3 = Dense(30)(middle1)
output3 = Dense(20)(output3)
output3 = Dense(10)(output3)
output3 = Dense(3)(output3)

#모델 선언
model = Model(inputs=[input1, input2], 
              outputs=[output1, output2, output3])
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train],[y1_train, y2_train, y3_train], epochs=100, batch_size=1,
          validation_split=0.2, verbose=1)

#4.평가 , 예측
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=1)

#loss: [3198.59716796875<전체로스(1번쨰loss+2번째loss 혹은 1번째 metrics+2번째 metrics)>, 2104.79638671875<1번째 loss값>, 1093.80078125<2번째 loss값>, 2104.79638671875<1번째 metrcis값>, 1093.80078125<2번째 metrics값>]
print("model.metrics_names : ", model.metrics_names)
print("loss:", loss)


y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])

print("=====================" )
print("y1_predict : \n", y1_predict )
print("=====================" )
print("y2_predict : \n", y2_predict )
print("=====================" )
print("y3_predict : \n", y3_predict )

#scikit learn
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y1_test, y1_predict, y3_predict))  #sqrt : 루트를 씌우기.

RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)
RMSE3 = RMSE(y3_test,y3_predict)
RMSE = (RMSE1+RMSE2+RMSE3)/3

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE :", RMSE)




'''
def RMSE2(y2_test, y2_predict):
    return np.sqrt(mean_squared_error(y2_test, y2_predict))  #sqrt : 루트를 씌우기.
print("RMSE2 : ", RMSE1(y2_test, y2_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))
'''
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)
r2 = (r2_1 + r2_2 + r2_3)/3


print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 :", r2)
