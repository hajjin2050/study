import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(dataset.DESCR)
# print(dataset.feature_names)
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size= 0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#2.MODEL


models = [KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in models:
    model = i



#3.COMPILE
    model.fit(x_train, y_train)
    print(f'\n{i}')
#4.EVALUATE
#result = model.evaluate(x,y)

    y_pred = model.predict(x_test)
    result = model.score(x_test,y_test)
    print("model.score:",result)
    r2 = r2_score(y_test,y_pred)
    print("r2_score:",r2)

# KNeighborsRegressor()
# model.score: 0.38507056834581477
# r2_score: 0.38507056834581477

# DecisionTreeRegressor()
# model.score: -0.3057500825266921
# r2_score: -0.3057500825266921

# RandomForestRegressor()
# model.score: 0.38660894141548297
# r2_score: 0.38660894141548297

#tensorflow
#acc: ???