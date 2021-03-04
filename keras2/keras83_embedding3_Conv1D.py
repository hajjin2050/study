from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요', '글세요', '별로에요', '생각보다 지루해요', '연기가 어색해요'
        ,'재미없어요','너무 재미없다','참 재밋네요','규현이가 잘 생길 뻔 했어요'
        ]

#긍정 1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5)#post 앞쪽을 0
print(pad_x)
print(pad_x.shape)  #(13,5)


print(np.unique(pad_x))
print(len(np.unique(pad_x)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5))#input_length = shape의 뒷자리(13,5)
# model.add(Embedding(28,11)) #flatten이 바로 안먹힌다 none때문에
# model.add(LSTM(32))
model.add(Conv1D(32,2))
model.add(Dense(1, activation='sigmoid'))
# model.add(Flatten())
# model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)