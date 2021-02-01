import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.decomposition import PCA

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

pca = PCA(n_components=9) #컬럼을 압축시킴
x2 = pca.fit_transform(x)
print(x2)
print(x2.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) #0.9479436357350414 컬럼을 다 합친경우
# 압축률 = sum(pca_EVR)

# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum:", cumsum)

d = np.argmax(cumsum>=0.95)+1
print("cumsum>=0.95", cumsum>=0.95)
print("d:", d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()