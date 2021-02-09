import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D

x_train = np.load("C:/data/image/brain/npy/keras66_train_x.npy")
y_train = np.load("C:/data/image/brain/npy/keras66_train_y.npy")
x_test = np.load("C:/data/image/brain/npy/keras66_test_x.npy")
y_test = np.load("C:/data/image/brain/npy/keras66_test_y.npy")

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#실습
#모델을 만들어라.

#MODELING
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state =44)


model = Sequential()
model.add(Conv2D(filters=400, kernel_size=(2,2), padding = 'same', strides=2
                        ,input_shape=(150, 150, 3)))
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
model.add(Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit_generator(
    x_train,y_train, epochs=100,
    validation_data =(x_val,y_val), validation_steps =4
)