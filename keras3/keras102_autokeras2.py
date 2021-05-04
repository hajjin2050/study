import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape)
print(x_train.shape)

x_train = x_train.reshape(60000, 28, 28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28,1).astype('float32')/255.

model = ak.ImageClassifier(
    overwrite = True,
    max_trials=2,
    loss = 'mse',
    metrics= ['mse']
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val-loss', patience=3, factor=0.5)
ck = ModelCheckpoint('./keras3/temp/', save_weight_only = True, monitor = 'val_loss',
                    verbose=0)

model.fit(x_train, y_train, epochs=10, validation_split=0.2,
            callbacks=[es,lr,ck] )

results = model.evaluate(x_test, y_test)

print(results)