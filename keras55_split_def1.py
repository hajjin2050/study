import numpy as np
import pandas as pd

data = np.load('../../data/npy/samsung3.npy')
x = data[:,:-1]
y = data[:,-1]
print(data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)

x = scaler.transform(x)
def split_xy3(data, time_steps, y_column) :
    x,y = list(), list()
    for i in range(len(data)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1

        if y_end_number > len(data) -1 :
            break
        tmp_x = data[i:x_end_number, : -1]
        tmp_y = data[x_end_number-1 : y_end_number-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(data, 4, 2)
print(x,"\n", y)
# [ 8.9800000e+04  9.1200000e+04  8.9100000e+04  8.9700000e+04
#   -9.9000000e-01  3.6068848e+07  4.6008070e+06]