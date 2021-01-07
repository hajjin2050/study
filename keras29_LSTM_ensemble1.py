import numpy as np
from numpy import array
#1.데이터
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,66,75])
x2_predict = array([65,75,85])# (3,) ->(1,3) ->(1,3,1)
                                    #Dense     #LSTM

print("x1.shape:",x1.shape) # (13, 3)
print("x2.shape:",x2.shape) # (13, 3)
print("y.shape:",y.shape)#(13,)
print("x1_predict.shape:",x1_predict.shape) # (3,)
print("x2_predict.shape:",x2_predict.shape) # (3,)


x1 = x1.reshape(x1.shape[0],x1.shape[1],1)#= 13, 3, 1
x2 = x2.reshape(x2.shape[0],x1.shape[1],1)
x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test, = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=311)
                                                                               #트레인과 테스트 해줄때는 그냥 x,y값 기준
from sklearn.model_selection import train_test_split
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, train_size=0.8, shuffle=True, random_state=311)
                                                                                  #트레인과 발 값 해줄때는 트레인기준
#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM, Input

#모델1
input1 = Input(shape =(3,1))
dense1 = LSTM(56, activation='relu')(input1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(45, activation='relu')(dense1)
dense1 = Dense(35, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

#모델2
input2 = Input(shape =(3,1))
dense2 = LSTM(50, activation='relu')(input2)
dense2 = Dense(35, activation='relu')(dense2)
dense2 = Dense(20, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

#모델변형 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle = Dense(18)(merge1)
middle = Dense(18)(middle)


output1 = Dense(30)(middle)
output1 = Dense(20)(output1)
output1 = Dense(15)(output1)
output1 = Dense(10)(output1)
output1 = Dense(1)(output1)


#모델 선언
model = Model(inputs=[input1, input2], 
              outputs=output1)
model.summary()
 
#3.COMPILE
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit([x1, x2], y, epochs=100, batch_size=3,validation_data=([x1_val, x2_val], y_val), callbacks=[early_stopping],verbose=3 )

#4.EVALUATE
loss = model.evaluate([x1, x2],y ,batch_size=16)
print("loss:",loss)


y_pred = model.predict([x1_predict, x2_predict])
print(y_pred)

#loss: 0.02820512466132641
# [[85.85441]]