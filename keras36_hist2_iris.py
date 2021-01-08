import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size= 0.8, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical #이게텐서플로우 1.0방식. 이것도 가능하긴 하다.

y = to_categorical(y)
y_train= to_categorical(y_train)
y_val= to_categorical(y_val)
y_test= to_categorical(y_test)

print(x.shape)#(150,4)
print(y.shape) # (105,3)
    


#2.MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(120, activation='relu',input_shape=(4,)))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(3, activation='softmax')) #다중분류=categorical_softmax, 이진분류=binary_crossentropy/sigmoid(activation)) / softmax로 나온 값은 모두 합치면 1이 된다. 이 중 가장 큰 값을 받은 애가 선택이 된다.

#3.COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stoppig = EarlyStopping(monitor='loss', patience=30, mode='auto')
hist = model.fit(x, y, epochs=500,  validation_data=(x_val, y_val), batch_size=8 , callbacks=[early_stoppig], verbose=3)

#4.EAVALUATE
loss = model.evaluate(x_test,y_test,batch_size=1)
print("loss :", loss)
# y_pred = model.predict(x[-5:-1])
# print("y_pred:", y_pred)
# print(y[-5:-1])


#그래프
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) #legend = >주석, 어떤그래프인지 
plt.show()







#loss : [1.1571476459503174, 0.6333333253860474, 0.3207857012748718]