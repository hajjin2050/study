import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col =0, header=0)

print(df)

print(df.shape) # (150,5)
print(df.info())

# df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

aaa = df.to_numpy() # pandas --> numpy
print(aaa)
print(type(aaa))
# 위와 동일
bbb= df.value_counts
print(bbb)
print(type(bbb))

np.save("../data/npy/iris_sklearn.npy")

#과제
#pandas의 loc iloc에 대해 정리