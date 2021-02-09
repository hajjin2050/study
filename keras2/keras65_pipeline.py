#목표 머신러닝과 DNN 묶어주기

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline,Pipeline
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 1. 데이터 / 전처리



y_train = to_categorical(y_train)  
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


# 2. 모델

def bulid_model(drop=0.5, optimizer='adam'):
    
    inputs = Input(shape=(28*28,), name='Input')
    x = Dense(512, activation='relu', name='Hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='Hidden2')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
    return model


def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam']
    dropout = [0.1,0.2,0.3]
    return {'pl__batch_size': batches, "pl":optimizers, "pl__drop":dropout}

hyperparameters = create_hyperparameters()
model2 = bulid_model()

model2 = KerasClassifier(build_fn=bulid_model, verbose=1, batch_size =32, epochs =10)
pipe = Pipeline([("scaler",MinMaxScaler()),("pl" ,model2)]) #(전처리 , 모델)
# 파이프 라인 위치!!!!(모델부분을 파이프라인으로 연결 (전처리-모델))
kfold = KFold(n_splits=2, random_state=42)
search = RandomizedSearchCV(pipe, hyperparameters, cv=kfold)

search.fit(x_train,y_train)

# print(search.best_params_)

print(search.best_score_)


acc = search.score(x_test, y_test)
print("최종 스코어 : ", acc)