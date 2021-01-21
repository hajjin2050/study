#(다입력, 다 : 1)

#1. 데이터
import numpy as np
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)
print(dataset)
print("dataset.shape:", dataset.shape)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number +y_column -1

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]
        tmp_y = dataset[x_end_number-1 : y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x, y = split_xy3(dataset, 3, 1)
print(x, "\n", y)
print(x.shape)
print(y.shape)