import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape)

# print(x_train.shape)
#1.data
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

#2.model
def build_model(drop = 0.5, optimizers='adam' ,act = 'relu', lr=0.01):
    optimizer = optimizers(lr=lr)
    inputs = input(shape = (28,28,1), name='input')
    x = Conv2D(128, 3, activation= act, padding = 'same', name ='conv1')(inputs)
    x = dropout(drop)(x)
    x = Conv2D(256, 3, activation= act, padding = 'same', name ='conv2')(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(256, 5, activation= act, padding = 'same', name ='conv3')(x)
    x = MaxPooling2D(5)(x)
    x = Flatten()(x)
    x = Dense(256, activation = act, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation =act, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax',name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer, metrics=['acc'],loss = 'categorical_crossentropy')

    return model

def create_hyperparameters():
    batches = [32, 64 ,128]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2]
    lr = [0.01, 0.05]
    act = ['relu, linear']
    return {"batch_size": batches, "optimizer":optimizers,
             "drop":dropout, 'act':act, 'lr':lr}
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

###########################################
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, ReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizers = Adam, act = 'relu', lr = 0.01, nodes = 256, layer = 1):
    optimizer = optimizers(lr = lr)
    inputs = Input(shape = (28,28,1), name = 'input')
    x = Conv2D(128, 3, activation = act, padding = 'same', name = 'conv1')(inputs)
    x = Dropout(drop)(x)
    for i in range(layer):
        x = Conv2D(nodes, 3, activation = act, padding = 'same', name = f'conv2{i}')(x)
    x = MaxPooling2D(3)(x)
    x = Conv2D(nodes, 5, activation = act, padding = 'same', name = 'conv3')(x)
    x = MaxPooling2D(5)(x)
    x = Flatten()(x)
    x = Dense(nodes, activation = act, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = act, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer, metrics = ['acc'], loss = 'categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [64, 128]
    optimizers = [Adam, RMSprop]
    lr = [0.01]
    dropout = [0.2, 0.3]
    act = ['relu']
    nodes = [256, 128]
    layer_num= [2, 3]

    return {'batch_size' : batches, 'optimizers' : optimizers, 'drop' : dropout, 'act': act, 'lr': lr, 'nodes' : nodes, 'layer' : layer_num}
hyperparameters = create_hyperparameter()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn= build_model, verbose = 1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

search.fit(x_train,y_train,verbose = 1)

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc)