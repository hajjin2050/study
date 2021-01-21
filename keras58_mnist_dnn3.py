import numpy as np
#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(10000, 28,28,1)                             # 실수형이라는 것을 빼도 인식한다.


y_train = x_train
y_tset = x_test

print(y_train.shape) #(60000, 28, 28, 1)
print(y_test.shape) #(10000,)



#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,Flatten

model = Sequential()
model.add(Dense(64, input_shape=(28,28, 1), activation='relu'))
# model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(80))
model.add(Dense(10))
model.add(Dense(1))
model.summary()


#3.COMPILE
model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'val_loss', patience=5)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감
model.fit(x_train, y_train , epochs=10 ,batch_size=16, verbose=1, callbacks=[es] )


#EVALUATE
loss, accuracy = model.evaluate(x_test,y_test, batch_size=64)
print("loss,accuracy:",loss,accuracy)
y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)


