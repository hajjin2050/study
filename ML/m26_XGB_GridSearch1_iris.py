#데이터 별로 5개 만든다.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from xgboost import XGBClassifier, plot_importance
import warnings
warnings.filterwarnings('ignore')


dataset = load_iris()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)

pca = PCA(n_components=154) #컬럼을 압축시킴
x2 = pca.fit_transform(x_train)
print(x2)
print(x2.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
pca = PCA()
pca.fit(x_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum:", cumsum)

d = np.argmax(cumsum>=0.95)+1
print("cumsum>=0.95", cumsum>=0.95)
print("d:", d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()
  
#dictionary
parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.01,0.03]
    ,"max_depth":[4,5,6]},
     {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
    ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1] },
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.3]
    ,"max_depth":[4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9] }
]


#2.MODEL
model = GridSearchCV(XGBClassifier(n_jobs = 8), parameters,cv=kfold) #나는 svc모델을 gridsearchcv방식으로 쓰겠다

scores = cross_val_score(model, x_train, y_train, cv=kfold)# cv = cross_val_score/ 모델과 데이터를 엮어줌

#3.COMPILE
model.fit(x_train, y_train)
#4.EVALUATE
y_pred = model.predict(x_test)
print("scores:", scores)


plot_importance(model)
plt.show()