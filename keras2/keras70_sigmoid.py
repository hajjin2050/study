import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5 , 0.1)
y = sigmoid(x)

print(x)
print(y)

#그림그리잣
plt.plot(x, y, 'k-')
plt.plot(1, 1, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()