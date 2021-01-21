import numpy as np

x_train = np.load('../data/npy/cifar100_x_train.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')

from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train =train_test_split(x_train, y_train, train_size= 0.6, shuffle=True, random_state = 112)
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
model.add(Dense(10, input_shape=(784, ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(160, activation='relu'))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(10000))
model.add(Dense(10, activation='softmax'))

# 모델이 끝난 지점에서 하면 모델만 저장된다.
model.save('../data/h5/k52_1_model1.h5')

#3.COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor = 'val_loss', patience=5)
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감


hist = model.fit(x_train, y_train , epochs=10,validation_data=(x_val, y_val) ,batch_size=16, verbose=1, callbacks=[es,cp] )

model.save('../../data/h5/k57_1_model2.h5')
model.save_weights('../../data/h5/k57_1_weight.h5')

#EVALUATE
loss, accuracy = model.evaluate(x_test,y_test, batch_size=64)
print("loss,accuracy:",loss,accuracy)
y_pred = model.predict(x_test)

x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# #MODELING
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
#                         ,input_shape=(32, 32, 3)))
# model.add(MaxPooling2D(pool_size = 2))
# model.add(Dropout(0.2))
# model.add(Conv2D(300, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(100, (2,2), padding='same'))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(90))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(100, activation='softmax'))

# model.save('../data/h5/k52_1_model1.h5')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpointM, ReduceLROnPlateau
modelpath ='../data/modelcheckpoint/k52_mnist_{epoch:02d}-{val_loss:4f}].hdf5'
redcuce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1) #factor => 3번까지 참고 개선이없으면 50프로 감축시키겠다
#k52_1_mnist_>>> => k52_1_MCK_.hdf5 이름을 바꿔줄것
es = EarlyStopping(monitor = 'loss', patience=10)
cp = ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감
model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train , epochs=10, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[es,cp] )

# model.save('.../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')
# model = load_model('../data/h5/k52_1_model2.h5')

#EVALUATE
# result = model1.evaluate(x_test,y_test, batch_size=64)
# print("model1_loss:",result[0])
# print("model1_accuracy:", result[1])
# y_pred = model1.predict(x_test)

# model.load_weights('../data/h5/k52_1_weight.h5')


# #EVALUATE -2
model = load_model('../../data/h5/k57_1_mnist_checkpoint.hdf5')
result = model.evaluate(x_test,y_test, batch_size=64)
print("model1_loss:",result[0])
print("model1_accuracy:", result[1])

# model2= load_model('../data/h5/k52_1_model2.h5')
# result2 = model2.evaluate(x_test, y_test, batch_size = 8)
# print("로드모델_loss:", result2[0])
# print("로드모델_accuracy:", result2[1])