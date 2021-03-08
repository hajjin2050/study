from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# x, y = load_boston(return_X_y=True)
dataset = load_boston()
x = dataset.data
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2 , shuffle=True, random_state = 66
)

#2.MODEL
model = XGBRegressor(n_estimators=10, learning_rate = 0.01, n_jobs=8)

#3.PRACTICE
model.fit(x_train, y_train ,verbose = 1 , eval_metric = ['rmse','logloss','mae']
           ,eval_set = [(x_train, y_train),(x_test, y_test)],
           early_stopping_rounds =10)

aaa = model.score(x_test,y_test)
# print(aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
# print("r2:", r2)

results = model.evals_result() # eval_metric(rmse)를 지표로 삼아 어떻게 측정값이 진행되는지 출력할수있음
# print(results) # model에서 설정한 n_estimators의 수 만큼 나옴


import joblib
# joblib.dump(model, "C:/data/xgb_save/m39.joblib.dat")
model.save_model("C:/data/xgb_save/m39.xgb.model")
model2 = XGBRegressor()
model2.load_model("C:/data/xgb_save/m39.xgb.model")

print('불러옴')
r22 = model2.score(x_test, y_test)
print('r22:', r22)
