import numpy as np
import pandas as pd

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
# 다 : 다
def split_train(dataset,x_row,x_col,y_row,y_col):
    x,y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i+x_row
        y_end_number = x_end_number + y_row

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
        return np.array(x), np.array(y)
x,y = split_train(dataset,1,3,2,3)
print(x)
print(y)