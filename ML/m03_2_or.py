from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1.DATA
x_data = [[0, 0],[1, 0],[0, 1],[1, 1]]
y_data = [0, 1, 1, 1]

#2. MODELING
model = LinearSVC()

#3. TRAIN
model.fit(x_data, y_data)

#4. EVALUATE
y_pred = model.predict(x_data)
print(x_data, "의 예측결과 :",y_pred)

result = model.score(x_data, y_data)
print("model.score:", result)

acc = accuracy_score(y_data, y_pred)
print('accuracy_score:', acc)