from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#(50000, 32, 32, 3) (50000, 1)#(10000, 32, 32, 3) (10000, 1)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle=True, random_state=66)

x_train  = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2], 3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2], 3)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2], 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


#MODELING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=400, kernel_size=2, padding = 'same', strides=2
                        ,input_shape=(1024, 3)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv1D(300, 2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(100, 2, padding='same'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(100, activation='sigmoid'))


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath ='../Data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f].hdf5'
es = EarlyStopping(monitor = 'loss', patience=5)
cp = ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train , epochs=10, batch_size=16, validation_data=(x_val, y_val), verbose=1, callbacks=[es,cp] )



#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)


#시각화

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  #판을 깔아줌

plt.subplot(2, 1, 1) #2,1 짜리 를 만들겠다 (2행 1열 중 첫번째)
plt.plot(hist.history['loss'], marker='.',c = 'red', label='loss') #로스라는 라벨을 빨간색으로 그려줄거다
plt.plot(hist.history['val_loss'], marker ='.', c = 'blue', label = ['val_loss']) # 발로스라는 라벨을 파란색으로 그려줄거다
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) #2,1 짜리 를 만들겠다 (2행 1열 중 두번째) 
plt.plot(hist.history['accuracy'], marker='.',c = 'red', label='accuracy') #로스라는 라벨을 빨간색으로 그려줄거다
plt.plot(hist.history['val_accuracy'], marker ='.', c = 'blue', label = ['val_accuracy']) # 발로스라는 라벨을 파란색으로 그려줄거다
plt.grid()#모눈종이 모양 

# plt.title('정확도')
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()