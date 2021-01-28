import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits = 5, shuffle = True)

allAlgorithms = all_estimators(type_filter='classifier')

for(name, algorithm) in allAlgorithms:
    try:#트라이문 안에서 예외(에러)가 발생하면 어떤행동을 취해라
        model = algorithm()
        
        scores = cross_val_score(model, x_train, y_train, cv=kfold) #cv에 숫자를 넣어도 자동으로 잘려짐
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률:', scores)
    except:
        # continue 
        print(name, '은 없는 놈!')

import sklearn#(SGbooster, LGBM없음)
print(sklearn.__version__)

# DecisionTreeClassifier 의 정답률: [1.         0.875      0.95833333 0.95833333 0.95833333]
# KNeighborsClassifier 의 정답률: [1.         1.         0.95833333 0.91666667 0.91666667]
# RandomForestClassifier 의 정답률: [0.91666667 0.95833333 0.95833333 1.         0.95833333]