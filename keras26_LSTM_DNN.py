#keras23_LSTM3_scale을  DNN 으로 코딩
#결과치 비교

#코딩하시오!! LSTM
#나는 80을 원하고있다

import numpy as np
#1.데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
    [20,30,40],[30,40,50],[40,50,60]
])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]) # (3,) >> (1,3,1)
x_pred = x_pred.reshape(1,3)

print(x.shape)
print(y.shape)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,train_size = 0.8,random_state = 121)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.2,random_state = 121)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)
scaler.transform(x_val)
# scaler.transform(x_pred)

x = x.reshape(13, 3)


#2.모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape=(3,),activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))



#3.컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=600, batch_size=5, verbose=3)

#4.평가 예측
loss = model.evaluate(x, y)
print("loss:",loss)

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3) 

result = model.predict(x_pred)
print("result:", result)
