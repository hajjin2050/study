import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier')

for(name, algorithm) in allAlgorithms:
    try:#트라이문 안에서 예외(에러)가 발생하면 어떤행동을 취해라
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률:',r2_score(y_test, y_pred))
    except:
        # continue 
        print(name, '은 없는 놈!')

import sklearn#(SGbooster, LGBM없음)
print(sklearn.__version__)