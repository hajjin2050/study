from helpers import *
import matplotlib.pyplot as plt
import os, glob, cv2, random
import seaborn as sns #데이터 시각화
import pandas as pd

base_path = 'dataset'

x, y = read_csv(os.path.join(base_path, 'dataset.csv'))

print(x.shape, y.shape)

#내가 봤을때 왼쪽 눈

plt.figure(figsize = (12,10))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(X[i].reshape((26,34)), cmap='gray')

sns.distplot(y, kde=False)

n_total = len(x)
x_result = np.empty((n_total, 26, 34, 1))

for i, x in enumerate(X):
    img = x.reshape((26, 34, 1))

    x_result[i] = img

from sklearn.model_selection import train_test_split

x_train, x_val , y_train, y_val = train_test_split(x_result, y , test_size = 0.1)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

np.save('C:/data/p_project/eye/x_train.npy', x_train)
np.save('C:/data/p_project/eye/y_train.npy', y_train)
np.save('C:/data/p_project/eye/x_val.npy', x_val)
np.save('C:/data/p_project/eye/y_val.npy', y_val)

plt.subplot(2, 1, 1) #subplot 은 한장에 여러개의 plot창을 띄움
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((26, 34)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_val[4]))
plt.imshow(x_val[4].reshape((26, 34)), cmap='gray')

sns.distplot(y_train, kde=False)

sns.distplot(y_val, kde=False)