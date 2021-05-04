import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import tensorflow as tf
import autokeras as ak

dataset = load_boston()
x  = dataset.data
y = dataset.target

#print(x.shape)(506, 13)
#print(y.shape)(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.6, random_state = 42
)


model = ak.StructuredDataRegressor(
    overwrite=True, max_trials=3
)

model.fit(x_train, y_train, epochs=1)

results = model.evaluate(x_test, y_test)

print(results)