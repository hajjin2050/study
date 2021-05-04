import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model = ak.ImageClassifier(max_trials=1, loss = 'mse', metrics=['acc'])
from tensorflow.keras.models import load_model
model = load_model('../keras3/temp/aaa.h5')
model.summary()    # 서머리가 먹는다.

best_model = load_model('./keras3/temp2/beset_aaa.h5')
model.fit(x_train,y_train,epochs=1)

###########################################

results = model.evaluate(x_test,y_test)
print(results)

best_results = best_model.evaluate(x_test, y_test)
print(best_results)