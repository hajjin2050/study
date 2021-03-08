from xgboost import XGBClassifier, XGBRFRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state =66
)

model = XGBRFRegressor(n_jobs = 8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2:", r2_score)

thresholds = np.sort(model.feature_importances_)
#sort => 낮은숫자부터 정렬할거임~(디폴트가 오름차순)
print(thresholds)

# [0.00388918 0.00719477 0.01237118 0.01373011 0.01635768 0.01780649
#  0.03177862 0.03296394 0.05993087 0.06438274 0.09072824 0.306246
#  0.34262007]

for thresh in thresholds :
    selection = SelectFromModel(model, threshold= thresh, prefit =True)
    #prehit => 알아보기 과제

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRFRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))

# (404, 13)
# Thresh=0.004, n=13, R2: 91.86%
# (404, 12)
# Thresh=0.007, n=12, R2: 91.65%
# (404, 11)
# Thresh=0.012, n=11, R2: 91.79%
# (404, 10)
# Thresh=0.014, n=10, R2: 91.70%
# (404, 9)
# Thresh=0.016, n=9, R2: 91.70%
# (404, 8)
# Thresh=0.018, n=8, R2: 91.45%
# (404, 7)
# Thresh=0.032, n=7, R2: 91.44%
# (404, 6)
# Thresh=0.033, n=6, R2: 91.54%
# (404, 5)
# Thresh=0.060, n=5, R2: 90.99%
# (404, 4)
# Thresh=0.064, n=4, R2: 90.91%
# (404, 3)
# Thresh=0.091, n=3, R2: 86.93%
# (404, 2)
# Thresh=0.306, n=2, R2: 79.71%
# (404, 1)
# Thresh=0.343, n=1, R2: 64.22%

print(model.intercept_)
print(model.coef_)
# AttributeError: Intercept (bias) is not defined for Booster type None