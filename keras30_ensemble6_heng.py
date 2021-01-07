import numpy as np
from numpy import array
#1.데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]])
            # [20,30,40],[30,40,50],[40,50,60]
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[80,90,100]
            ,[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y1 = array([4,5,6,7,8,9,10,11,12,13])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,66,75])
x2_predict = array([65,75,85])# (3,) ->(1,3) ->(1,3,1)
                                    #Dense     #LSTM

print("x1.shape:",x1.shape)
print("x2.shape:",x2.shape)
# print("y.shape:",y.shape)


x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x1.shape[1],1)

print("x1.shape:",x1.shape)
print("x2.shape:",x2.shape)


# x2 = np.transpose(3,)

#2.모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,LSTM,Input

#모델1
input1 = Input(shape =(3,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)

#모델2
input2 = Input(shape =(3,1))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)

#모델변형 / concatenate
from tensorflow.keras.layers import concatenate,Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])

merge1 = Dense(10)(merge1)

#모델 분기1
output1 = Dense(30)(merge1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

#모델 분기2
output2 = Dense(30)(merge1)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)



#모델 선언
model = Model(inputs=[input1, input2], 
              outputs=[output1, output2])
model.summary()
 
#3.COMPILE
model.compile(loss='mse', optimizer='adam')
model.fit([x1, x2], [y1,y2]
, epochs=100, verbose=1 )

#4.EVALUATE
result = model.evaluate([x1,x2], [y1,y2])
print("result:",result)

x1_pred = x1_predict.reshape(1,3,1)
x2_pred = x2_predict.reshape(1, 3, 1)

result = model.predict([x1_pred , x2_pred])
print(result)

