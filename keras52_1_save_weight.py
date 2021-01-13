import numpy as np

#1. 데이터
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x 전처리
x_train = x_train.reshape(60000, 28*28* 1)
x_test = x_test.reshape(10000, 28*28* 1)                             # 실수형이라는 것을 빼도 인식한다.
 
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# y 리쉐잎
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# y 벡터화 OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
hot = OneHotEncoder()
hot.fit(y_train)
y_train = hot.transform(y_train).toarray()
y_val = hot.transform(y_val).toarray()
y_test = hot.transform(y_test).toarray()


#2. 모델 구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(200, input_shape=(784, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(160, activation='relu'))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(10, activation='softmax'))

# 모델이 끝난 지점에서 하면 모델만 저장된다.
model.save('../data/h5/k52_1_model1.h5')

#3.COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor = 'val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감


hist = model.fit(x_train, y_train , epochs=10,validation_data=(x_val, y_val), batch_size=16, verbose=1, callbacks=[es,cp] )

model.save('../data/h5/k52_1_model2.h5')
model.save_weights('../data/h5/k52_1_weight.h5')

#EVALUATE
loss, accuracy = model.evaluate(x_test,y_test, batch_size=64)
print("loss,accuracy:",loss,accuracy)
y_pred = model.predict(x_test)