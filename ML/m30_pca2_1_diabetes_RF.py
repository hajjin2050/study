#랜포로 모델링하시오!!!
#전처리
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)

pca = PCA(n_components=2) #컬럼을 압축시킴
x2 = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x2, datasets.target, train_size=0.8, random_state=44
)
# print(x2)
# print(x2.shape)

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

model = RandomForestClassifier(max_depth = 4)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("acc:", acc)

# acc: acc: 0.8666666666666667 (3)
# acc: 0.956140350877193 (9)