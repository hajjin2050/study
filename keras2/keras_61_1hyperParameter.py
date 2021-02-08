import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape)

# print(x_train.shape)
#1.data
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2.MODEL
def build_model(drop = 0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation = 'relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation ='relu', name = 'hidden1')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name='hidden1')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation = 'softmax', name='outputs')
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size": batches, "optimizer":optimizers,
             "drop":dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model2 = KerasClassifier(build_fn = build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv =3)

search.fit(x_train, y_train, verbose=1)

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print("final_score:", acc)
