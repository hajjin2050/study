import numpy as np
import autokeras as ak
from sklearn.datasets import load_breast_cancer
#1.DATA
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print('x.shape:',x.shape)   #(569,30)
print('y.shape:',y.shape)   #(569,)
# print(x[:5])
# print(y)

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(
    x, y, train_size = 0.8, shuffle=True, random_state = 66)


model = ak.StructuredDataClassifier(
    overwrite=True, max_trials=3
)

model.fit(x_train, y_train, epochs=1)

results = model.evaluate(x_test, y_test)

print(results)