import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x +6  #2차함수
x = np.linspace(-1, 6 ,100) #-1 부터 6까지 100개를 집어넣겠다.
y = f(x)

#그림그리잣
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()