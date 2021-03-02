#실습
#cifar10으로 vgg16넣어서 만들것
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10


(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

vgg16 = VGG16(weights='imagenet',include_top = False, input_shape=(32, 32, 3))
# print(vgg16.weights)

vgg16.trainable=False

vgg16.summary()
print(len(vgg16.weights)) #26
print(len(vgg16.trainable_weights))#0


model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=0.2,verbose=1)

print("acc:", accuracy)

print("그냥 가중치의 수:", len(model.weights))  #32
print("동결하기 ?? 훈련되는 가중치의 수:", len(model.trainable_weights))  #6 #프리즌~

###############요기 하단때문에 파일 분리했다.


import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)