#RandomSearch, GS 와 pipeline을 엮어라
#모델은 RandomForest
#gridsearch와 pipeline을 이어줌
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)

parameters = [
    {'randomforestclassifier__n_estimators': [100,200]},
    {'randomforestclassifier__max_depth' : [6, 8]},
    {'randomforestclassifier__min_samples_leaf': [3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_split' : [2, 3, 5,10]},
]
#RandomForestClassifier()=> 안에 변수들을 확인할수있음.
scale = [MinMaxScaler(), StandardScaler()]
search = [RandomizedSearchCV, GridSearchCV] # 함수로 들어가는거라 ()생략

for i in scale:
    pipe = make_pipeline(i, RandomForestClassifier()) #[]를 넣어서 오류가 났었음
    for j in search:
        model = j(pipe, parameters, cv=5)
        model.fit(x_train, y_train)
        print(f'score{i}_{j.__name__}', model.score(x_test, y_test))

# scoreMinMaxScaler()_RandomizedSearchCV 0.9666666666666667
# scoreMinMaxScaler()_GridSearchCV 0.9666666666666667
# scoreStandardScaler()_RandomizedSearchCV 0.9666666666666667
# scoreStandardScaler()_GridSearchCV 1.0
