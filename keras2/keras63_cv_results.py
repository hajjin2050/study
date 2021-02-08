# epchs 100 적용
# validation_split, callback 적용
# early_stopping 5 적용
#Reduce LR 3 적용


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, ReLU, LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#1. 데이터 / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28).astype('float32')/255.
x_test = x_test.reshape(10000,28,28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5, optimizers = Adam, act = 'relu', lr = 0.01, nodes = 256, layer = 1):
    optimizer = optimizers(lr = lr)
    inputs = Input(shape = (28,28), name = 'input')
    x = LSTM(128, activation = act, name = 'LSTM1')(inputs)
    x = Dropout(drop)(x)
    for i in range(layer):
        x = Dense(nodes, activation = act, name = f'LSTM_{i}')(x)
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
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor= 'loss', patience=5, mode='auto')
redcuce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

hyperparameters = create_hyperparameter()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn= build_model, verbose = 1)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

search.fit(x_train,y_train,verbose = 1, epochs=100,validation_split=0.2, callbacks=[early_stopping,redcuce_lr])

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc)
########################3333
print(search.cv_results_)