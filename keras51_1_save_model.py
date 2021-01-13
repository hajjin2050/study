import numpy as np

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
 #1.DATA
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  #(10000, 28, 28) (10000,)


# X reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train =train_test_split(x_train, y_train, train_size= 0.6, shuffle=True, random_state = 112)


#Y reshape
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#MODELING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
                        ,input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(300, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), padding='same'))
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
model.add(Dense(10, activation='softmax'))

model.save('../data/h5/k51_1_model1.h5')


model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k51_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# d(정수형)으로 10의 자리까지 /f(float) 실수형으로 소수 4번째까지 

es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min')
cp = ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감


hist = model.fit(x_train, y_train , epochs=100, batch_size=16, validation_data=(x_val, y_val), verbose=1, callbacks=[es,cp] )

model.save('../data/h5/k51_1_model2.h5')


#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)
