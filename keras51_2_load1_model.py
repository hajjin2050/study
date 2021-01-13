import numpy as np

x_train = np.load('../data/npy/cifar100_x_train.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')

from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train =train_test_split(x_train, y_train, train_size= 0.6, shuffle=True, random_state = 112)


x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#MODELING
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
model = load_model('../data/h5/k51_1_model1.h5')

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath ='../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f].hdf5'
es = EarlyStopping(monitor = 'loss', patience=5)
cp = ModelCheckpoint(filepath='modelpath', monitor='val_loss', save_best_only=True, mode='auto') #filepath -> 그 지점의 weigh값이 들어감
model.compile(loss = 'mse', optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train, y_train , epochs=10, batch_size=16, validation_data=(x_val, y_val), verbose=1, callbacks=[es,cp] )

# model.save('../data/h5/k51_1_model2.h5')


#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)
