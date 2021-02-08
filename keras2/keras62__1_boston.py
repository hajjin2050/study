import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

dataset = load_boston()
x  = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 44)
# print(x_train.shape)(404, 13)
# print(x_test.shape)(102, 13(102, 13)
# print(y_test.shape)(102,)
# print(y_train.shape)(404,)

#1.data
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2.MODEL
def build_model(drop = 0.5, optimizer='adam'):
    inputs = Input(shape=(13,), name='input')
    x = Dense(512, activation = 'relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation ='relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(1,activation = 'relu', name='outputs')(x)
    model = Model(inputs = inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='mse')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size": batches, "optimizer":optimizers,
             "drop":dropout}
hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor= 'loss', patience=5, mode='auto')
redcuce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
model2 = KerasRegressor(build_fn = build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv =3)

search.fit(x_train,y_train,verbose = 1, epochs=100,validation_split=0.2, callbacks=[early_stopping,redcuce_lr])

print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
acc = search.score(x_test, y_test)
print("final_score:", acc)

# RandomizedSearchCV
# final_score: 0.9803922176361084