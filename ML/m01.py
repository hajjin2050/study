import numpy as np
import matpoltlib.pyplot as plt

x = np.aragne(0, 10, 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()