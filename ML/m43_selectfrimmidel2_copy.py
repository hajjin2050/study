from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

x, y = load_boston(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

xgb = XGBRegressor()

parameters = {
    'max_depth' : [2, 4, 6, -1],
    'min_child_weight' : [1, 2, 4, -1],
    'eta' : [0.3, 0.1, 0.01, 0.5]
}
model = RandomizedSearchCV(xgb, param_distributions= parameters, cv = 5)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

thresholds = np.sort(model.best_estimator_.feature_importances_) # 이렇게 하면 fi 값들이 정렬되어 나온다!

tmp = 0
tmp2 = [0,0]
for thresh in thresholds:
    
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    if score > tmp :
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = select_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
    print('Best Score so far : ', tmp)
    print('Best Threshold : ', tmp2[0])

print('=========================================================================================')
print(f'Best Threshold : {tmp2[0]}, n = {tmp2[1]}')

selection = SelectFromModel(model.best_estimator_, threshold = tmp2[0], prefit = True)

select_x_train = selection.transform(x_train)

selection_model = RandomizedSearchCV(xgb, parameters, cv =5)
selection_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print('=========================================================================================')
print(f'최종 R2 score : {score*100}%, n = {tmp2[1]}일때!!')