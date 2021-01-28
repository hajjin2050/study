#SVC를 사용하여 어큐러씨 쓰코어  1.0
#머신러닝 방식의 딥러닝 

from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.DATA
x_data = [[0, 0],[1, 0],[0, 1],[1, 1]]
y_data = [0, 1, 1, 0]

#2. MODELING
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

#3. TRAIN
model.compile(loss='binary_crossentropy', optimizer ='adam', metrics =['acc'])
model.fit(x_data, y_data, batch_size = 1 , epochs = 100)

#4. EVALUATE
y_pred = model.predict(x_data)
print(x_data, "의 예측결과 :",y_pred)

result = model.evaluate(x_data, y_data)
print("model.score:", result[1])

# acc = accuracy_score(y_data, y_pred)
# print('accuracy_score:', acc)