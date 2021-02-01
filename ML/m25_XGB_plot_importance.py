from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, plot_importance

#1.DATA
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=44)
import datetime

date_now1 = datetime.datetime.now()
date_time = date_now1.strftime("%m월%d일_%H시%M분%S초")
print("start time: ",date_time)  

# model = GradientBoostingClassifier(max_depth = 4)
model = XGBClassifier(n_jobs= 8) #코어를 몇개 사용하겠다 (-1 은 전부다 사용 / 현재는 8개가 최대)
 # n_jobs 1 4 8 -1 compare

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

fi = model.feature_importances_
fi = pd.DataFrame(fi).quantile(q=0.3)
fi = fi.to_numpy()
print(fi)
print("before acc:", acc)

df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
cl = df.columns
new_data=[]
for i in range(len(cl)):
    if model.feature_importances_[i] > fi:
        new_data.append(df.iloc[:,i])
new_data = pd.concat(new_data, axis = 1)

new_data1 = new_data.to_numpy()

x2_train, x2_test, y2_train, y2_test = train_test_split(new_data1, dataset.target, train_size= 0.8 , random_state=44)
model2 = GradientBoostingClassifier(max_depth = 4)
model2.fit(x2_train, y2_train)
acc2 =model2.score(x2_test, y2_test)
print("after acc:" ,acc2)

date_now2 = datetime.datetime.now()
date_time = date_now2.strftime("%m월%d일_%H시%M분%S초")
print("End time: ",date_time)
print("걸린시간 : ",(date_now2-date_now1))

import matplotlib.pyplot as plt
import numpy as np
'''
def plot_feature_importances_dataset(model):
    n_features = new_data1.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), new_data.columns)
    plt.xlabel("Feature importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model2)
'''


plot_importance(model)
plt.show()

#n_jobs = -1
# 걸린시간 :  0:00:00.288256

#n_jobs = 1
# 걸린시간 :  0:00:00.284243

#n_jobs = 4
# 걸린시간 :  0:00:00.280284

#n_jobs=8
# 걸린시간 :  0:00:00.273272