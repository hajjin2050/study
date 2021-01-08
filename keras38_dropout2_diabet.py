# 실습 19_1~earlystopping까지 만들기 (다른 데이터셋)

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

# print(x[:5])
# print(y[:10])
print(x.shape, y.shape)

print(np.max(x), np.min(y))
print(dataset.feature_names)
print(dataset.DESCR)


from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size =0.6 , random_state=66
)

#2.MODEL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape=(10,))
dense1 = Dense(200,activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(150,activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(100,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(80,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(50,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(30,activation='relu')(dense1)
dropout1 = Dropout(0.2)(dense1)
output = Dense(1)(dense1)

model = Model(inputs = input1 ,outputs= output)
model.summary()

#3.COMPILE
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=3)

#4.EVALUATE
loss, mae = model.evaluate(x_test, y_test, batch_size=8, )
print("model.metrics_names : ", model.metrics_names)
print("loss,mae:", loss, mae)


y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred) :
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print("R2 :", r2_score(y_test, y_pred))

# origin
# loss,mae: 2921.361083984375 42.85511779785156
# RMSE : 54.04961843324407
# R2 : 0.49844822361671093


# drop out
# loss,mae: 3038.4462890625 43.453086853027344
# RMSE : 55.1221017981575
# R2 : 0.47834659586054606