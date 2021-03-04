from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
#{'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6} 
# 빈도수가 높은거대로 순서대로 

x = token.texts_to_sequences([text])
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size) #6
x = to_categorical(x)

print(x)
print(x.shape)
