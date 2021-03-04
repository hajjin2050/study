from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)
'''
print(x_train[0])
print("=======================")
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

print("뉴스기사 최대길이:", max(len(l)for l in x_train))
print("뉴스기사 최대길이:", sum(map(len, x_train))/len(x_train))

plt.hist([len(s) for s in x_train], bins=50)

#y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("y분포 :",dict(zip(unique_elements, counts_elements)))
print("===========================")

plt.hist(y_train, bins=46)
plt.show()

word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print("=====================================")

#키와 밸류를 교체
index_to_word ={}
for key, value in word_to_index.items():
    index_to_word[value] = key
#키 밸류 교환후
print(index_to_word)
print(index_to_word[1])
print(index_to_word[30979])
print(len(index_to_word))

# x_train[0]
print(x_train[0])
print(''.join([index_to_word[index(x)]for index in x_train[0]]))

#y카테고리 갯수 출력
category = np.max(y_train) + 1
print("y 카테고리 갯수 : ", category) #46

#y의 유니크한 값출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
'''
################################전처리########################
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen = 100, padding='pre')
x_test = pad_sequences(x_test, maxlen = 100, padding='pre')
print(x_train.shape, x_test.shape)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, Flatten

model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))
model.add(Embedding(10000,64))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose =1)

results = model.evaluate(x_test, y_test)

print("loss:", results[0])
print("acc:", results[1])