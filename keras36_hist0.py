import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1,101))
size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1 ):
        subset = seq[i : (i+size)]
        aaa.append([ subset])#이렇게 item for item in을 빼도 되는데 리스트가 한번 더 씌워져서 나온다.
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x= dataset[:,0:4]
y = dataset[:,-1]
print(x.shape, y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)

#2.MODEL
from tensorflow.keras.models import load_model
model = load_model('/model/svae_keras35.h5')
model.add(Dense(5, name='moonkeras1'))
model.add(Dense(1, name='moonkeras2'))

from tensorflow.keras.callbacks import EearlyStopping
es = EearlyStopping(monitor='loss', patiednce=10, mode='auto')

#3.COMPILE
model.compile(loss = 'mse', optimizer ='adam')
hist = model.fit(x, y, epochs=100, batch_size=32,verbose=3, validation_split = 0.2, callbacks=[es] )
print(hist)
print(hist.history.key())


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