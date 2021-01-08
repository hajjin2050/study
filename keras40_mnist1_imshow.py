#인공지능계의 hello world라 불리는 mnist!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  #(600000, 28, 28) (600000,)
print(x_test.shape, y_test.shape)    #(100000, 28, 28) (100000,)

print(x_train[0])
print("y_train[0]:",y_train[0])
print(x_train.shape)

plt.imshow(x_train[0], 'gray')
plt.show()