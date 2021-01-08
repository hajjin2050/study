#인공지능계의 hello world라 불리는 mnist!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(600000, 28, 28) (600000,)
print(x_test.shape, y_test.shape)    #(100000, 28, 28) (100000,)

print(x_train[0])
print("y_train[0]:",y_train[0]) #y_train[0]: 5
print(x_train.shape) 

# plt.imshow(x_train[0], 'gray')
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# # OneHotEencoding
# from sklearn.preprocessing import OneHotEncoder
# OneHotEncoder.fit(y_test)
# y_test_onehot = OneHotEncoder.transform(x_test).toarray()
# y_test = np.argmax(y_test_onehot, axis=1).reshape(-1,1)
# print(y_test)

#MODELING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
                        ,input_shape=(28,28,1)))
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
model.add(Dense(1))

model.compile(loss = 'mae', optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train , epochs=10, batch_size=64, verbose=3 )


#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)
print(y_test[:10])
print(y_pred[:10])

'''
y_test[:10] = (?,?,?,?,?,?,?,?,?,?,?)
y_pred[:10] = (?,?,?,?,?,?,?,?,?,?,?)'''