#keras21_cancer1.py를 다중분류로 코딩하시오.
import numpy as np
from sklearn.datasets import load_iris
#1.DATA
datasets = load_iris()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
y = np.reshape(y, (150,1))

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
#이 내용을 아래 sklearn 으로 바꿈

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

print(y[:5])
print(y.shape)


from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)

from sklearn.model_selection import train_test_split
x_val, x_train, y_val, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)



#2.MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30,activation = 'relu',input_shape=(4,))) #'relu' - > 0~무한으로 수렴
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(3, activation='softmax'))

#3.compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #loss='binary_crossentropy'->이진분류일때 사용!!
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=earlystopping)



#4.EVALUATE
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss:",loss)

y_pred = model.predict(x)
# print('y_pred:',y_pred)
# print("y[-5:-1]:",y[-5:-1])


# loss: [0.07601557672023773, 0.9666666388511658]