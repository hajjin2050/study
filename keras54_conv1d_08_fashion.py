from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train =  x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])


#MODELING
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
                        ,input_shape=(28,28)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv1D(300, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(100, (2,2), padding='same'))
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

model.compile(loss = 'mse', optimizer='adam',metrics=['mae'])
model.fit(x_train, y_train , epochs=10, batch_size=64, verbose=3 )


#EVALUATE
loss, mae = model.evaluate(x_test,y_test, batch_size=64)
print("loss,mae:",loss,mae)
y_pred = model.predict(x_test)